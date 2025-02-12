# Physics-Informed 3D Surrogate for Elastic Fields in Polycrystals
# Training script

## Basic imports below

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader , TensorDataset
import matplotlib.pyplot as plt
import psutil
import time
from mpl_toolkits.axes_grid1 import ImageGrid
import csv
import gc
import itertools
import yaml
from mytools import *


## Loading config file
config=load_config("train_config.yaml")

## Getting output folder location
outdir=config['plotting']['results_folder']

## Creating output folder and subfolders
os.system('mkdir -p '+outdir)
os.system('mkdir -p '+outdir+'out_nn_train')
os.system('mkdir -p '+outdir+'out_nn_train'+'/sigma')
os.system('mkdir -p '+outdir+'out_fft_train')
os.system('mkdir -p '+outdir+'out_error_train')
os.system('mkdir -p '+outdir+'out_nn_val')
os.system('mkdir -p '+outdir+'out_fft_val')
os.system('mkdir -p '+outdir+'out_error_val')
os.system('mkdir -p '+outdir+'losses')
os.system('mkdir -p '+outdir+'performance_train')
os.system('mkdir -p '+outdir+'performance_val')
os.system('mkdir -p '+outdir+'models')

## Copying input files into output folder
os.system('cp *.py '+outdir)
os.system('cp *.yaml '+outdir)  

## Fixing random seed for reproducibility
torch.manual_seed(config['model']['random_seed'])

## Enable garbage collector for saving on RAM
gc.enable()

## Running configurations
if config['run']['try_cuda']==True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    f = open(outdir+"info_cuda.txt", "w")
    f.write("PyTorch version: "+str(torch.__version__)+"\n")
    f.write("CUDA IS "+str(torch.cuda.is_available()))
    f.close()

## FOR CUDA INTERACTIVE MACHINE ONLY
if config['run']['try_interactive']==True:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

## Creating file for recording maximum RAM consumption
f = open(outdir+"memory.txt", "w")
f.write("GPU memory consumption (in bytes): "+str(torch.cuda.max_memory_allocated(device=None)))
f.close()

## Getting folder of input data (datasets for quaternion and stress fields)
inputdir=config['dataset']['path']

## Reading input data
quat_dataset=np.load(inputdir+config['dataset']['name_x_data'])
sigma_fft_dataset=np.load(inputdir+config['dataset']['name_y_data'])

## Reading number of samples for training and validation
n_train_samples=config['train']['n_samples']
n_val_samples=config['val']['n_samples']

## Separating the dataset into separate tensors for training and validation
quat_train=torch.from_numpy(quat_dataset[0:n_train_samples]).to(gettensorprecision(precision)).to(device)
sigma_fft_train=torch.from_numpy(sigma_fft_dataset[0:n_train_samples]).to(gettensorprecision(precision)).to(device)
quat_val=torch.from_numpy(quat_dataset[n_train_samples:n_train_samples+n_val_samples]).to(gettensorprecision(precision)).to(device)
sigma_fft_val=torch.from_numpy(sigma_fft_dataset[n_train_samples:n_train_samples+n_val_samples]).to(gettensorprecision(precision)).to(device)

## Manually freeing some RAM
del quat_dataset
del sigma_fft_dataset
torch.cuda.empty_cache()

## Plotting 2D crops of ground truth stress fields
plotcropsvector('sig_xx_fft',sigma_fft_train[config['train']['sample_for_plotting']:config['train']['sample_for_plotting']+1].to('cpu'),
                config['plotting']['stress_component'],
                outdir+'out_fft_train/')
plotcropsvector('sig_xx_fft',sigma_fft_val[config['val']['sample_for_plotting']:config['val']['sample_for_plotting']+1].to('cpu'),
                config['plotting']['stress_component'], 
                outdir+'out_fft_val/')

## Initializing kernels for finite differences via convolutions
kernel_x_grad,kernel_y_grad,kernel_z_grad=fd_kernels()
kernel_x_div,kernel_y_div,kernel_z_div=bd_kernels()

kernels_grad=torch.cat(( 
    kernel_x_grad.reshape(1,*kernel_x_grad.shape),
    kernel_y_grad.reshape(1,*kernel_y_grad.shape),
    kernel_z_grad.reshape(1,*kernel_z_grad.shape)
    ),0).to(device)

kernels_div=torch.cat((
    kernel_x_div.reshape(1,*kernel_x_div.shape),
    kernel_y_div.reshape(1,*kernel_y_div.shape),
    kernel_z_div.reshape(1,*kernel_z_div.shape)
    ),0).to(device)

## Boundary conditions
# Material properties
try: 
    if config['BCs']['material']=='gamma_tial':
        C0_2=np.array([[183.0e3,74.0e3,74.0e3,0,0,0],
            [74.0e3,183.0e3,74.0e3,0,0,0],
            [74.0e3,74.0e3,178.0e3,0,0,0],
            [0,0,0,105.0e3,0,0],
            [0,0,0,0,105.0e3,0],
            [0,0,0,0,0,78.0e3]],dtype=precision)
except:
    print('Material name not recognized, please try again.')

C0_4=torch.tensor(sec2fourth(C0_2),dtype=gettensorprecision(precision),requires_grad=False).to(device)

# Prescribed macroscopic strain tensor
eps_macro=torch.tensor([
                [config['BCs']['macro_strain'][0],config['BCs']['macro_strain'][1],config['BCs']['macro_strain'][2]],
                [config['BCs']['macro_strain'][1],config['BCs']['macro_strain'][3],config['BCs']['macro_strain'][4]],
                [config['BCs']['macro_strain'][2],config['BCs']['macro_strain'][4],config['BCs']['macro_strain'][5]]],dtype=gettensorprecision(precision),requires_grad=False).to(device)

## Architecture parameters
nf1=int(config['model']['nf1'])
nf2=int(config['model']['nf2'])
nf3=int(config['model']['nf3'])
nf4=int(config['model']['nf4'])
## Number of epochs (times the entire dataset will be "seen")
n_epochs=config['train']['epochs']
## Early stopping: stop the model when it no longer performs better on validation data than the previous epochs
early_stopping=config['optimizer']['early_stopping']
## Patience: how many epoch to wait before stopping
patience=config['optimizer']['patience']
## Minimum delta: minimum performance gain to characterize a significant improvement
min_delta=config['optimizer']['min_delta']
## Initial learning rate
lr=float(config['optimizer']['initial_lr'])
## Plotting frequency
plot_freq=config['plotting']['plot_freq']

## Initializing model
model=physicsinformed3Dresnet_custom(kernels_grad,C0_4,eps_macro,nf1,nf2,nf3,nf4)

## Calculating and saving number of parameters
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
f = open(outdir+"parameters.txt", "w")
f.write("Number of parameters: "+str(params)+"\n")
f.close()

## Activating model parallelism if needed
model= nn.DataParallel(model)
## Sending model to GPU (if activated)
model.to(device)

## Initializing loss function
if config['optimizer']['supervised']==True:
    loss_fn_train=suploss(sigma_fft_train)
    loss_fn_val=suploss(sigma_fft_val)
else:
    loss_fn=divloss(kernels_div)

## Initializing MAE layers
train_error_layer=pred_error(sigma_fft_train)
val_error_layer=pred_error(sigma_fft_val)

## Initializing optimizer
if config['optimizer']['name']=='sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
elif config['optimizer']['name']=='adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
else:
    print('Optimizer not recognized. Please try again.')

## Initializing early stopper
early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

## Initializing empty lists
train_loss_list=[]
val_loss_list=[]
train_error_list=[]
val_error_list=[]

## Training loop:
memory=0
for epoch in range(n_epochs):
    
    # Activate training mode (gradient calculation)

    model.train()
    running_train_loss=0
    running_train_error=0
    
    for sample in range(n_train_samples):
    
        # Forward pass
        sig_nn_train  = model(quat_train[sample:sample+1])  ## OUTPUT STILL IN GPU
        if config['optimizer']['supervised']==True:
            train_loss = loss_fn_train(sig_nn_train)  ## OUTPUT STILL IN GPU
        else:
            train_loss = loss_fn(sig_nn_train)  ## OUTPUT STILL IN GPU

        train_error=train_error_layer(sig_nn_train)

        
        running_train_loss=running_train_loss+train_loss.to('cpu').item()
        running_train_error=running_train_error+train_error.to('cpu').item()
    
        # Backward and optimize
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    
    # Activate eval mode and deactivate gradient calculation
    model.eval()
    running_val_loss=0
    running_val_error=0
    
    for sample in range(n_val_samples):
        with torch.no_grad():

            sig_nn_val  = model(quat_val[sample:sample+1])  ## OUTPUT STILL IN GPU
            if config['optimizer']['supervised']==True:
                val_loss = loss_fn_val(sig_nn_val)  ## OUTPUT STILL IN GPU
            else:
                val_loss = loss_fn(sig_nn_val)  ## OUTPUT STILL IN GPU
            val_error=val_error_layer(sig_nn_val)

        running_val_loss=running_val_loss+val_loss.to('cpu').item()
        running_val_error=running_val_error+val_error.to('cpu').item()
        
    print("Epoch %d: Training loss = %.6f, Validation loss = %.6f , Training error = %.5f, Validation error = %.5f" % (epoch, running_train_loss, running_val_loss, running_train_error, running_val_error ))

    if torch.cuda.max_memory_allocated(device=None)>memory:
        f = open(outdir+"memory.txt", "w")
        f.write("GPU memory consumption (in bytes): "+str(torch.cuda.max_memory_allocated(device=None)))
        f.close()
        memory=torch.cuda.max_memory_allocated(device=None)

    train_loss_list.append(running_train_loss/n_train_samples)
    val_loss_list.append(running_val_loss/n_val_samples)
    train_error_list.append(running_train_error/n_train_samples)
    val_error_list.append(running_val_error/n_val_samples)
    
    # Saving loss function values
    np.savetxt(outdir+'losses/'+'train_loss.csv',train_loss_list,delimiter =", ",fmt ='%.4f')
    np.savetxt(outdir+'losses/'+"val_loss.csv",val_loss_list,delimiter =", ",fmt ='%.4f')
    np.savetxt(outdir+'losses/'+'train_error.csv',train_error_list,delimiter =", ",fmt ='%.4f')
    np.savetxt(outdir+'losses/'+"val_error.csv",val_error_list,delimiter =", ",fmt ='%.4f')
    
    if epoch>0 and (running_val_loss/n_val_samples)<np.mean(val_loss_list[epoch-1:epoch]):
        torch.save(model, outdir+'models/bestmodel_inner_depth_'+str(width)
                    +'_n_resblocks_'+str(depth)
                    +'.pt')

    if (epoch+1)%config['plotting']['plot_freq']==0:
        
        # Plotting results
        fig=plt.figure()
        ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
        handle_leg_train,=ax.plot(train_loss_list, label='Training')
        handle_leg_val,=ax.plot(val_loss_list, label='Validation')
        ax.set_ylim(0,max(max(train_loss_list,val_loss_list)))
        ax.set_xlim(0,len(train_loss_list)-1)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend(handles=[handle_leg_train, handle_leg_val])
        fig.savefig(outdir+'losses/loss1.png')
        plt.close('all')
        
        fig=plt.figure()
        ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
        handle_leg_train,=ax.plot(train_loss_list, label='Training')
        handle_leg_val,=ax.plot(val_loss_list, label='Validation')
        ax.set_ylim(0,5*min(min(train_loss_list,val_loss_list)))
        ax.set_xlim(0,len(train_loss_list)-1)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend(handles=[handle_leg_train, handle_leg_val])
        fig.savefig(outdir+'losses/loss2.png')
        plt.close('all')
        
        fig=plt.figure()
        ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
        handle_leg_train,=ax.plot(train_loss_list, label='Training')
        handle_leg_val,=ax.plot(val_loss_list, label='Validation')
        ax.set_ylim(0,1)
        ax.set_xlim(0,len(train_loss_list)-1)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend(handles=[handle_leg_train, handle_leg_val])
        fig.savefig(outdir+'losses/loss3.png')
        plt.close('all')
        
        fig=plt.figure()
        ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
        handle_leg_train,=ax.plot(train_loss_list, label='Training')
        handle_leg_val,=ax.plot(val_loss_list, label='Validation')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend(handles=[handle_leg_train, handle_leg_val])
        fig.savefig(outdir+'losses/loss4.png')
        plt.close('all')
        
        fig=plt.figure()
        ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
        handle_leg_train,=ax.plot(train_error_list, label='Training')
        handle_leg_val,=ax.plot(val_error_list, label='Validation')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Mean absolute error (NN vs FFT)')
        ax.legend(handles=[handle_leg_train, handle_leg_val])
        fig.savefig(outdir+'losses/error1.png')
        plt.close('all')
        
        os.system('mkdir -p '+outdir+'out_nn_train/sigma/'+'epoch_'+str(epoch))
        os.system('mkdir -p '+outdir+'out_nn_val/sigma/'+'epoch_'+str(epoch))
        os.system('mkdir -p '+outdir+'out_error_train/'+'epoch_'+str(epoch))
        os.system('mkdir -p '+outdir+'out_error_val/'+'epoch_'+str(epoch))

        firstsample_sig_nn_train=model(quat_train[0:1]).to('cpu')  ## 0 is for sigma
        firstsample_sig_nn_val=model(quat_val[0:1]).to('cpu')    ## 0 is for sigma
        
        plotcropstensor('sig_xx_nn',firstsample_sig_nn_train, 'xx', outdir+'out_nn_train/sigma/'+'epoch_'+str(epoch)+'/')
        plotcropstensor('sig_xx_nn',firstsample_sig_nn_val, 'xx', outdir+'out_nn_val/sigma/'+'epoch_'+str(epoch)+'/')
        plotcropserror('error_sig_xx',sigma_fft_train[0:1].to('cpu'),firstsample_sig_nn_train,'xx',outdir+'out_error_train/'+'epoch_'+str(epoch)+'/')
        plotcropserror('error_sig_xx',sigma_fft_val[0:1].to('cpu'),firstsample_sig_nn_val,'xx',outdir+'out_error_val/'+'epoch_'+str(epoch)+'/')

        ## Correlation diagrams for grain-wise averages
        performance_scatterplot(quat_train[0:1].to('cpu'),sigma_fft_train[0:1,0:1,:,:,:].to('cpu'),firstsample_sig_nn_train[0:1,0:1,0,:,:,:],outdir+'performance_train/','sigxx_epoch_'+str(epoch))
        performance_scatterplot(quat_val[0:1].to('cpu'),sigma_fft_val[0:1,0:1,:,:,:].to('cpu'),firstsample_sig_nn_val[0:1,0:1,0,:,:,:],outdir+'performance_val/','sigxx_epoch_'+str(epoch))
        
    if early_stopping==True:
        if config['optimizer']['monitor']=='train':
            if early_stopper.early_stop(running_train_loss):             
                break
        elif config['optimizer']['monitor']=='val':
            if early_stopper.early_stop(running_val_loss):             
                break
        else:
            print('Early stopper monitor not recognized')

