import os
import math
import numpy as np
from numpy import array_equal
from numpy import loadtxt
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
import sklearn
from sklearn.neighbors import KernelDensity

precision="float"

CONFIG_PATH = "./"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

def gettensorprecision(precision):
    if precision=="double":
        return torch.double
    else:
        return torch.float

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

def convert_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
      
    return "%d:%02d:%02d" % (hour, minutes, seconds)

def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if array_equal(elem, myarr)), False)

## FORWARD DIFFERENCES
def fd_kernels():
    kernel_x_fd=torch.tensor([[
                    [0,0,0],  
                    [0,0,0],
                    [0,0,0]],
                    [
                    [0,0,0],
                    [0,-1,0],
                    [0,0,0]],
                    [
                    [0,0,0],
                    [0,1,0],
                    [0,0,0]]],dtype=gettensorprecision(precision),device=device,requires_grad=False)
    kernel_y_fd=torch.tensor([[
                    [0,0,0],
                    [0,0,0],
                    [0,0,0]],
                    [
                    [0,0,0],
                    [0,-1,0],
                    [0,1,0]],
                    [
                    [0,0,0],
                    [0,0,0],
                    [0,0,0]]],dtype=gettensorprecision(precision),device=device,requires_grad=False)
    kernel_z_fd=torch.tensor([[
                    [0,0,0],
                    [0,0,0],
                    [0,0,0]],
                    [
                    [0,0,0],
                    [0,-1,1],
                    [0,0,0]],
                    [
                    [0,0,0],
                    [0,0,0],
                    [0,0,0]]],dtype=gettensorprecision(precision),device=device,requires_grad=False)
    return kernel_x_fd,kernel_y_fd,kernel_z_fd

def bd_kernels():
    kernel_x_bd=torch.tensor([[
                    [0,0,0],  
                    [0,-1,0],
                    [0,0,0]],
                    [
                    [0,0,0],
                    [0,1,0],
                    [0,0,0]],
                    [
                    [0,0,0],
                    [0,0,0],
                    [0,0,0]]],dtype=gettensorprecision(precision),device=device,requires_grad=False)
    kernel_y_bd=torch.tensor([[
                    [0,0,0],
                    [0,0,0],
                    [0,0,0]],
                    [
                    [0,-1,0],
                    [0,1,0],
                    [0,0,0]],
                    [
                    [0,0,0],
                    [0,0,0],
                    [0,0,0]]],dtype=gettensorprecision(precision),device=device,requires_grad=False)
    kernel_z_bd=torch.tensor([[
                    [0,0,0],
                    [0,0,0],
                    [0,0,0]],
                    [
                    [0,0,0],
                    [-1,1,0],
                    [0,0,0]],
                    [
                    [0,0,0],
                    [0,0,0],
                    [0,0,0]]],dtype=gettensorprecision(precision),device=device,requires_grad=False)
    return kernel_x_bd,kernel_y_bd,kernel_z_bd


class kernel_diff(nn.Module):
    def __init__(self,kernel):
        super().__init__()
        # self.kernel=nn.Parameter(kernel.repeat(3, 1, 1, 1), requires_grad=False)
        self.kernel=nn.Parameter(kernel, requires_grad=False)
        # print(self.kernel.size())
        
    def forward(self,x):
        x=F.pad(x,(1,1,1,1,1,1), "circular")
        # print(x.size())
        kernel=self.kernel.reshape(1,1,*self.kernel.shape)
        # print(kernel.size())
        
        return F.conv3d(x,kernel,padding='valid')


class allkernelsdiff(nn.Module):
    def __init__(self,allkernels):
        super().__init__()
        self.allkernels=nn.Parameter(allkernels, requires_grad=False)
    
    def forward(self,x):
        x=F.pad(x,(1,1,1,1,1,1), "circular")
        # k1=self.allkernels[0,:,:,:]
        # k1=k1.reshape(1,1,*k1.shape)
        # k2=self.allkernels[1,:,:,:]
        # k2=k2.reshape(1,1,*k2.shape)
        # k3=self.allkernels[2,:,:,:]
        # k3=k3.reshape(1,1,*k3.shape)
        k1=self.allkernels[0,:,:,:].reshape(1,1,*self.allkernels[0,:,:,:].shape)
        k2=self.allkernels[1,:,:,:].reshape(1,1,*self.allkernels[1,:,:,:].shape)
        k3=self.allkernels[2,:,:,:].reshape(1,1,*self.allkernels[2,:,:,:].shape)
        
        return torch.cat((
            F.conv3d(x,k1,padding='valid'),
            F.conv3d(x,k2,padding='valid'),
            F.conv3d(x,k3,padding='valid'))
            ,1)


class gradsym_minibatch(nn.Module):
    def __init__(self,kernels):
        super().__init__()
        self.alldifflayer=allkernelsdiff(kernels)
    def forward(self,u):
        # a=self.difflayer(u[:,0:1,:,:,:])[:,0,:,:,:]
        # print(a.size())
        ## SHAPE OF EACH "TERM": 1,32,32,32
        derivatives_u0=self.alldifflayer(u[:,0:1,:,:,:])
        derivatives_u1=self.alldifflayer(u[:,1:2,:,:,:])
        derivatives_u2=self.alldifflayer(u[:,2:3,:,:,:])
        xx=derivatives_u0[:,0,:,:,:]
        xy=torch.mul(0.5,torch.add(derivatives_u0[:,1,:,:,:],derivatives_u1[:,0,:,:,:]))
        xz=torch.mul(0.5,torch.add(derivatives_u0[:,2,:,:,:],derivatives_u2[:,0,:,:,:]))
        yy=derivatives_u1[:,1,:,:,:]
        yz=torch.mul(0.5,torch.add(derivatives_u1[:,2,:,:,:],derivatives_u2[:,1,:,:,:]))
        zz=derivatives_u2[:,2,:,:,:]
        firstline=torch.unsqueeze(torch.cat((xx,xy,xz),0),1)
        secondline=torch.unsqueeze(torch.cat((xy,yy,yz),0),1)
        thirdline=torch.unsqueeze(torch.cat((xz,yz,zz),0),1)
        assembled=torch.cat((firstline,secondline,thirdline),1)
        matrix=torch.unsqueeze(assembled, 0)
        return matrix

class div_minibatch(nn.Module):
    def __init__(self,kernels):
        super().__init__()
        self.diffk1_layer=kernel_diff(kernels[0,:,:,:])
        self.diffk2_layer=kernel_diff(kernels[1,:,:,:])
        self.diffk3_layer=kernel_diff(kernels[2,:,:,:])
    def forward(self,sig):
        dSxxdk1=self.diffk1_layer(sig[:,0:1,0,:,:,:])
        dSxydk2=self.diffk2_layer(sig[:,0:1,1,:,:,:])
        dSxzdk3=self.diffk3_layer(sig[:,0:1,2,:,:,:])
        dSxydk1=self.diffk1_layer(sig[:,1:2,0,:,:,:])
        dSyydk2=self.diffk2_layer(sig[:,1:2,1,:,:,:])
        dSyzdk3=self.diffk3_layer(sig[:,1:2,2,:,:,:])
        dSxzdk1=self.diffk1_layer(sig[:,2:3,0,:,:,:])
        dSyzdk2=self.diffk2_layer(sig[:,2:3,1,:,:,:])
        dSzzdk3=self.diffk3_layer(sig[:,2:3,2,:,:,:])
        firstelement=torch.add(dSxxdk1,torch.add(dSxydk2,dSxzdk3))
        secondelement=torch.add(dSxydk1,torch.add(dSyydk2,dSyzdk3))
        thirdelement=torch.add(dSxzdk1,torch.add(dSyzdk2,dSzzdk3))
        return torch.cat((firstelement,secondelement,thirdelement),dim=1)

class div_sig_vector(nn.Module):
    def __init__(self,kernels):
        super().__init__()
        self.diffk1_layer=kernel_diff(kernels[0,:,:,:])
        self.diffk2_layer=kernel_diff(kernels[1,:,:,:])
        self.diffk3_layer=kernel_diff(kernels[2,:,:,:])
    def forward(self,sig_fft):
        dSxxdk1=self.diffk1_layer(sig_fft[:,0:1,:,:,:])
        dSxydk2=self.diffk2_layer(sig_fft[:,1:2,:,:,:])
        dSxzdk3=self.diffk3_layer(sig_fft[:,2:3,:,:,:])
        dSxydk1=self.diffk1_layer(sig_fft[:,1:2,:,:,:])
        dSyydk2=self.diffk2_layer(sig_fft[:,3:4,:,:,:])
        dSyzdk3=self.diffk3_layer(sig_fft[:,4:5,:,:,:])
        dSxzdk1=self.diffk1_layer(sig_fft[:,2:3,:,:,:])
        dSyzdk2=self.diffk2_layer(sig_fft[:,4:5,:,:,:])
        dSzzdk3=self.diffk3_layer(sig_fft[:,5:6,:,:,:])
        firstelement=torch.add(dSxxdk1,torch.add(dSxydk2,dSxzdk3))
        secondelement=torch.add(dSxydk1,torch.add(dSyydk2,dSyzdk3))
        thirdelement=torch.add(dSxzdk1,torch.add(dSyzdk2,dSzzdk3))
        return torch.cat((firstelement,secondelement,thirdelement),dim=1)



class gradsym_batch(nn.Module):
    def __init__(self,kernels):
        super().__init__()
        self.minibatchlayer=gradsym_minibatch(kernels)
    def forward(self,ubatch):
        out=self.minibatchlayer(ubatch[0:1])
        if ubatch.size()[0]==1:
            return out
        else:
            for i in range(ubatch.size()[0]):
                out=torch.cat(out,self.minibatchlayer(ubatch[i:i+1]),dim=0)
            return out
    
class div_batch(nn.Module):
    def __init__(self,kernels):
        super().__init__()
        self.minibatchlayer=div_minibatch(kernels)
    def forward(self,sig):
        out=self.minibatchlayer(sig[0:1])
        if sig.size()[0]==1:
            return out
        else:
            for i in range(sig.size()[0]):
                out=torch.cat(out,self.minibatchlayer(sig[i:i+1]),dim=0)
            return out

## Takes two pairs of ijkl indices and returns 1 (equivalent) or 0 (not equivalent)
## according to both major and minor symmetries
def symm_check(i1,j1,k1,l1,i2,j2,k2,l2):
    if (i1==i2 and j1==j2 and k1==k2 and l1==l2):
        return 1
    elif (i1==j2 and j1==i2 and k1==k2 and l1==l2):
        return 1
    elif (i1==j2 and j1==i2 and k1==l2 and l1==k2):
        return 1
    elif(i1==i2 and j1==j2 and k1==l2 and l1==k2):
        return 1
    elif(i1==k2 and j1==l2 and k1==i2 and l1==j2):
        return 1
    else:
        return 0

## Transfer a pair of indices i,j (inside a 4th order tensor for example)
## into Voigt notation (components from 1 to 6, here 0 to 5)
def index_transf(i,j):
    if (i==0 and j==0):
        return 0
    elif (i==1 and j==1):
        return 1
    elif (i==2 and j==2):
        return 2
    elif (i==1 and j==2) or (i==2 and j==1):
        return 3
    elif (i==0 and j==2) or (i==2 and j==0):
        return 4
    elif (i==0 and j==1) or (i==1 and j==0):
        return 5


def sec2fourth(C2):
    dummy=np.zeros(shape=[3,3,3,3],dtype=precision)
    dummy[0,0,0,0]=C2[0,0];
    dummy[1,1,1,1]=C2[1,1];
    dummy[2,2,2,2]=C2[2,2];
    dummy[0,0,1,1]=C2[0,1];
    dummy[0,0,2,2]=C2[0,2];
    dummy[1,1,2,2]=C2[1,2];
    dummy[1,2,1,2]=C2[3,3];
    dummy[0,2,0,2]=C2[4,4];
    dummy[0,1,0,1]=C2[5,5];
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            for o in range(3):
                                for p in range(3):
                                    if symm_check(i,j,k,l,m,n,o,p)==1:
                                        dummy[i,j,k,l]=\
                                        max(dummy[i,j,k,l],dummy[m,n,o,p])
    return dummy


## DON'T USE FOR DEEP LEARNING (OPERATIONS WILL BE DONE WITH THE FOURTH ORDER TENSORS)
def fourth2sec(C4):
    C2=np.zeros(shape=[6,6],dtype=precision)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    a=index_transf(i,j)
                    b=index_transf(k,l)
                    C2[a,b]=C4[i,j,k,l]
    return C2

def fourth2sec_batch(C4batch):
    C4batchnp=C4batch.numpy()
    C2batchnp=np.zeros(np.shape(C4batchnp))
    for batchsize in range(np.size(C4batchnp)[0]):
        for i1 in range(np.size(C4batchnp)[-3]):
            for i2 in range(np.size(C4batchnp)[-2]):
                for i3 in range(np.size(C4batchnp)[-1]):
                    C2batchnp[batchsize,:,:,i1,i2,i3]=fourth2sec(C4batchnp[batchsize,:,:,:,:,i1,i2,i3])
    return torch.from_numpy(C2batchnp)
################

def eu2quat(euler,P=-1):
    phi1=euler[0]
    phi2=euler[2]
    bigphi=euler[1]
    sigma=0.5*(phi1+phi2)
    delta=0.5*(phi1-phi2)
    c=np.cos(0.5*bigphi)
    s=np.sin(0.5*bigphi)
    q0=c*np.cos(sigma)
    q=np.array([q0,-P*s*np.cos(delta),-P*s*np.sin(delta),-P*c*np.sin(sigma)])
    if q0<0:
        return -q
    else:
        return q
    
def qu2om(qu,P=-1):
    q0=qu[0]
    q1=qu[1]
    q2=qu[2]
    q3=qu[3]
    qbar=(q0**2)-((q1**2)+(q2**2)+(q3**2))
    
    a11=qbar+(2*(q1**2))
    a12=2*((q1*q2)-(P*q0*q3))
    a13=2*((q1*q3)+(P*q0*q2))
    
    a21=2*((q1*q2)+(P*q0*q3))
    a22=qbar+(2*(q2**2))
    a23=2*((q2*q3)-(P*q0*q1))
    
    a31=2*((q1*q3)-(P*q0*q2))
    a32=2*((q2*q3)+(P*q0*q1))
    a33=qbar+(2*(q3**2))
    matrix=np.array([[a11,a12,a13],
            [a21,a22,a23],
            [a31,a32,a33]])
    matrix=np.transpose(matrix)
    return matrix

class qu2rotmats(nn.Module):
    def __init__(self,P=-1):
        super().__init__()
        self.P=P
    def forward(self,quattensorbatch):
        q0=quattensorbatch[:,0:1,:,:,:]
        q1=quattensorbatch[:,1:2,:,:,:]
        q2=quattensorbatch[:,2:3,:,:,:]
        q3=quattensorbatch[:,3:4,:,:,:]
        qbar=torch.subtract(torch.mul(q0,q0), torch.add(torch.mul(q1,q1),torch.add(torch.mul(q2,q2),torch.mul(q3,q3))))
        alpha11=torch.unsqueeze(torch.add(qbar,torch.mul(2,torch.mul(q1,q1))), dim=1)
        alpha12=torch.unsqueeze(torch.mul(2,torch.subtract(torch.mul(q1,q2),torch.mul(self.P,torch.mul(q0,q3)))), dim=1)
        alpha13=torch.unsqueeze(torch.mul(2,torch.add(torch.mul(q1,q3),torch.mul(self.P,torch.mul(q0,q2)))), dim=1)

        alpha21=torch.unsqueeze(torch.mul(2,torch.add(torch.mul(q1,q2),torch.mul(self.P,torch.mul(q0,q3)))), dim=1)
        alpha22=torch.unsqueeze(torch.add(qbar,torch.mul(2,torch.mul(q2,q2))), dim=1)
        alpha23=torch.unsqueeze(torch.mul(2,torch.subtract(torch.mul(q2,q3),torch.mul(self.P,torch.mul(q0,q1)))), dim=1)
        
        alpha31=torch.unsqueeze(torch.mul(2,torch.subtract(torch.mul(q1,q3),torch.mul(self.P,torch.mul(q0,q2)))), dim=1)
        alpha32=torch.unsqueeze(torch.mul(2,torch.add(torch.mul(q2,q3),torch.mul(self.P,torch.mul(q0,q1)))), dim=1)
        alpha33=torch.unsqueeze(torch.add(qbar,torch.mul(2,torch.mul(q3,q3))), dim=1)
        
        alpha_line1=torch.cat((alpha11,alpha12,alpha13),dim=2)
        alpha_line2=torch.cat((alpha21,alpha22,alpha23),dim=2)
        alpha_line3=torch.cat((alpha31,alpha32,alpha33),dim=2)
        alpha_matrix=torch.cat((alpha_line1,alpha_line2,alpha_line3),dim=1)
        alpha_matrix=torch.transpose(alpha_matrix, 1, 2)
        return alpha_matrix


class alphaC0_42C_4(nn.Module):
    def __init__(self,C0_4):
        super().__init__()
        self.C0_4=C0_4
    def forward(self,alphatensor):
        onestensor=torch.ones(size=(alphatensor.size()[0],3,3,3,3,alphatensor.size()[3],alphatensor.size()[4],alphatensor.size()[5]),dtype=gettensorprecision(precision))
        C0_4tensor=torch.einsum('bijklxyz,ijkl->bijklxyz',onestensor,self.C0_4)
        Gtensor=torch.einsum('bmixyz,bnjxyz,bokxyz,bplxyz->bijklmnopxyz',alphatensor,alphatensor,alphatensor,alphatensor)
        # C0_4tensor.to(gettensorprecision(precision))
        C_4tensor=torch.einsum('bijklmnopxyz,bmnopxyz->bijklxyz',Gtensor,C0_4tensor)
        return C_4tensor


class eps_per2eps(nn.Module):
    def __init__(self,E_macro):
        super().__init__()
        self.E_macro=E_macro
    def forward(self,eps_per):
        onestensor=torch.ones(size=eps_per.size(),dtype=gettensorprecision(precision))
        E_macro_tensor=torch.einsum('bijxyz,ij->bijxyz',onestensor,self.E_macro)
        # print(E_macro_tensor.size())
        # print(E_macro_tensor[0,:,:,0,0,0])
        eps=torch.add(eps_per,E_macro_tensor)
        # print(eps[0,:,:,0,0,0])
        return eps


class epsC42sig(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,C4,eps):
        return torch.einsum('bijklxyz,bklxyz->bijxyz',C4,eps)


class ConvInstNormAct(nn.Module):
    def __init__(self,in_channels,out_channels,activation,instancenorm=True,padding='circular',conv_biases=True):
        super().__init__()
        self.modules=[]
        self.modules.append(nn.Conv3d(in_channels, 
                      out_channels, 
                      # kernel_size=(3,3,3),
                      kernel_size=(5,5,5),
                      stride=(1,1,1),
                      # padding=(1,1,1),
                      padding=(2,2,2),
                      padding_mode=padding,
                      bias=conv_biases,
                      dtype=gettensorprecision(precision)
                      ))
        if instancenorm==True:
            self.modules.append(nn.InstanceNorm3d(out_channels,dtype=gettensorprecision(precision)))
        self.modules.append(activation)
        self.apply_layer=nn.Sequential(*self.modules)
    def forward(self,x):
        return self.apply_layer(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, CBA, middle_act,instancenorm=True,padding='circular',conv_biases=True):
        super().__init__()
        self.ops=[]
        self.ops.append(CBA)
        self.ops.append(
            nn.Conv3d(out_channels, 
                      out_channels, 
                      # kernel_size=(3,3,3),
                      kernel_size=(5,5,5),
                      stride=(1,1,1),
                      # padding=(1,1,1),
                      padding=(2,2,2),
                      padding_mode=padding,
                      bias=conv_biases,
                      dtype=gettensorprecision(precision)
                      ))
        if instancenorm==True:
            self.ops.append(nn.InstanceNorm3d(out_channels,dtype=gettensorprecision(precision)))
        self.ops=nn.Sequential(*self.ops)
        self.act=nn.Sequential(
            middle_act
            )
    def forward(self,x):
        return self.act(torch.add(self.ops(x),x))


class resnet3D(nn.Module):
    def __init__(self, in_channels, out_channels,inner_depth,number_resblocks,activation_start,activation_middle,activation_end,instancenorm=True):
        super().__init__()
        self.modules=[]
        self.modules.append(ConvInstNormAct(in_channels, inner_depth, activation_start,instancenorm=instancenorm))
        for i in range(number_resblocks):
            self.modules.append(ResidualBlock(inner_depth,
                                              inner_depth,
                                              ConvInstNormAct(inner_depth, inner_depth, activation_middle),
                                              activation_middle,
                                              instancenorm=instancenorm))
        self.modules.append(ConvInstNormAct(inner_depth,out_channels, activation_end,instancenorm=instancenorm))
        self.apply_model=nn.Sequential(*self.modules)
    def forward(self,x):
        return self.apply_model(x)



class physicsinformed3Dresnet(nn.Module):
    def __init__(self,allkernels,C0_4,eps_macro,inner_depth,n_resblocks):
        super().__init__()
        self.allkernels=allkernels
        self.C0_4=C0_4
        self.eps_macro=eps_macro
        self.quat2alpha_layer=qu2rotmats()
        self.alphaC0_42C_4_layer=alphaC0_42C_4(self.C0_4)
        self.gradsym_batch_layer=gradsym_batch(self.allkernels)
        self.eps_per2eps_layer=eps_per2eps(self.eps_macro)
        self.epsC42sig_layer=epsC42sig()
        self.resnet=resnet3D(4,
                3,
                inner_depth,
                n_resblocks,
                nn.LeakyReLU(),
                nn.LeakyReLU(),
                nn.Identity(),
                instancenorm=True)

    def forward(self,quat):
        alpha=self.quat2alpha_layer(quat)
        C4=self.alphaC0_42C_4_layer(alpha)
        u_per=self.resnet(quat)
        eps_per=self.gradsym_batch_layer(u_per)
        eps=self.eps_per2eps_layer(eps_per)
        sig=self.epsC42sig_layer(C4,eps)
        return sig

class divloss(nn.Module):
    def __init__(self,allkernels):
        super().__init__()
        self.allkernels=allkernels
        self.div_layer=div_batch(self.allkernels)

    def forward(self, sig):
        div=self.div_layer(sig)
        return torch.mul(1e-7,torch.mean(torch.norm(div)))
    
class divloss_standardized(nn.Module):
    def __init__(self,allkernels):
        super().__init__()
        self.allkernels=allkernels
        self.div_layer=div_batch(self.allkernels)

    def forward(self, sig):
        sigxx=sig[:,0,0,:,:,:]
        sigxy=sig[:,0,1,:,:,:]
        sigxz=sig[:,0,2,:,:,:]
        sigyy=sig[:,1,1,:,:,:]
        sigyz=sig[:,1,2,:,:,:]
        sigzz=sig[:,2,2,:,:,:]
        std_sigxx=(sigxx-torch.mean(sigxx))*(1/torch.std(sigxx))
        std_sigxy=(sigxy-torch.mean(sigxy))*(1/torch.std(sigxy))
        std_sigxz=(sigxz-torch.mean(sigxz))*(1/torch.std(sigxz))
        std_sigyy=(sigyy-torch.mean(sigyy))*(1/torch.std(sigyy))
        std_sigyz=(sigyz-torch.mean(sigyz))*(1/torch.std(sigyz))
        std_sigzz=(sigzz-torch.mean(sigzz))*(1/torch.std(sigzz))
        line1=torch.unsqueeze(torch.cat((std_sigxx,std_sigxy,std_sigxz),dim=0),dim=0)
        line2=torch.unsqueeze(torch.cat((std_sigxy,std_sigyy,std_sigyz),dim=0),dim=0)
        line3=torch.unsqueeze(torch.cat((std_sigxz,std_sigyz,std_sigzz),dim=0),dim=0)
        std_sig=torch.unsqueeze(torch.cat((line1,line2,line3),dim=0),dim=0)
        div=self.div_layer(std_sig)
        return torch.mul(1,torch.mean(torch.norm(div)))

class suploss(nn.Module):
    def __init__(self,sigtensorfft):
        super().__init__()
        self.sigfft=sigtensorfft
    
    def forward(self,sig_nn):
        term1=torch.square(torch.sub(self.sigfft[0,0,:,:,:],sig_nn[0,0,0,:,:,:]))
        term2=torch.square(torch.sub(self.sigfft[0,1,:,:,:],sig_nn[0,0,1,:,:,:]))
        term3=torch.square(torch.sub(self.sigfft[0,2,:,:,:],sig_nn[0,0,2,:,:,:]))
        term4=torch.square(torch.sub(self.sigfft[0,3,:,:,:],sig_nn[0,1,1,:,:,:]))
        term5=torch.square(torch.sub(self.sigfft[0,4,:,:,:],sig_nn[0,1,2,:,:,:]))
        term6=torch.square(torch.sub(self.sigfft[0,5,:,:,:],sig_nn[0,2,2,:,:,:]))
        return torch.mul(1e-10,torch.mean(term1)+torch.mean(term2)+torch.mean(term3)+torch.mean(term4)+torch.mean(term5)+torch.mean(term6))

class pred_error(nn.Module):
    def __init__(self,sigtensorfft):
        super().__init__()
        self.sigfft=sigtensorfft
    
    def forward(self,sig_nn):
        term1=torch.abs(torch.sub(self.sigfft[0,0,:,:,:],sig_nn[0,0,0,:,:,:]))
        term2=torch.abs(torch.sub(self.sigfft[0,1,:,:,:],sig_nn[0,0,1,:,:,:]))
        term3=torch.abs(torch.sub(self.sigfft[0,2,:,:,:],sig_nn[0,0,2,:,:,:]))
        term4=torch.abs(torch.sub(self.sigfft[0,3,:,:,:],sig_nn[0,1,1,:,:,:]))
        term5=torch.abs(torch.sub(self.sigfft[0,4,:,:,:],sig_nn[0,1,2,:,:,:]))
        term6=torch.abs(torch.sub(self.sigfft[0,5,:,:,:],sig_nn[0,2,2,:,:,:]))
        return (torch.mean(term1)+torch.mean(term2)+torch.mean(term3)+torch.mean(term4)+torch.mean(term5)+torch.mean(term6))/6

class hybridloss(nn.Module):
    def __init__(self,allkernels,sig_fft):
        super().__init__()
        self.sig_fft=sig_fft
        self.allkernels=allkernels
        self.divlosslayer = divloss(allkernels)
        self.suplosslayer = suploss(sig_fft)
    def forward(self,sig_nn):
        print("Supervised loss %e" % self.suplosslayer(sig_nn).mean())
        print("Self-supervised loss %e" % self.divlosslayer(sig_nn).mean())
        return torch.add(self.divlosslayer(sig_nn),self.suplosslayer(sig_nn))


def plotcropstensor(name,sigtensor_nn,component,outdir):
    if component=='xx':
        dummy=sigtensor_nn[0,0,0,:,:,:].detach().numpy()
    elif component=='xy':
        dummy=sigtensor_nn[0,0,1,:,:,:].detach().numpy()
    elif component=='xz':
        dummy=sigtensor_nn[0,0,2,:,:,:].detach().numpy()
    elif component=='yy':
        dummy=sigtensor_nn[0,1,1,:,:,:].detach().numpy()
    elif component=='yz':
        dummy=sigtensor_nn[0,1,2,:,:,:].detach().numpy()
    elif component=='zz':
        dummy=sigtensor_nn[0,2,2,:,:,:].detach().numpy()
    for i in range(np.shape(dummy)[0]):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.imshow(dummy[i,:,:])
        plt.colorbar()
        plt.savefig(outdir+name+'_crop_'+'x_'+str(i)+'.png')
        plt.close("all")


def plotcropsvector(name,sigvector_fft,component,outdir):
    if component=='xx':
        dummy=sigvector_fft[0,0,:,:,:]
    elif component=='xy':
        dummy=sigvector_fft[0,1,:,:,:]
    elif component=='xz':
        dummy=sigvector_fft[0,2,:,:,:]
    elif component=='yy':
        dummy=sigvector_fft[0,3,:,:,:]
    elif component=='yz':
        dummy=sigvector_fft[0,4,:,:,:]
    elif component=='zz':
        dummy=sigvector_fft[0,5,:,:,:]
    for i in range(np.shape(dummy)[0]):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.imshow(dummy[i,:,:])
        plt.colorbar()
        plt.savefig(outdir+name+'_crop_'+'x_'+str(i)+'.png')
        plt.close("all")


def plotcropserror(name,sigvector_fft,sigtensor_nn,component,outdir):
    if component=='xx':
        gt=sigvector_fft[0,0,:,:,:]
        pred=sigtensor_nn[0,0,0,:,:,:].detach().numpy()
    elif component=='xy':
        gt=sigvector_fft[0,1,:,:,:]
        pred=sigtensor_nn[0,0,1,:,:,:].detach().numpy()
    elif component=='xz':
        gt=sigvector_fft[0,2,:,:,:]
        pred=sigtensor_nn[0,0,2,:,:,:].detach().numpy()
    elif component=='yy':
        gt=sigvector_fft[0,3,:,:,:]
        pred=sigtensor_nn[0,1,1,:,:,:].detach().numpy()
    elif component=='yz':
        gt=sigvector_fft[0,4,:,:,:]
        pred=sigtensor_nn[0,1,2,:,:,:].detach().numpy()
    elif component=='zz':
        gt=sigvector_fft[0,5,:,:,:]
        pred=sigtensor_nn[0,2,2,:,:,:].detach().numpy()
    err=np.abs(gt-pred)
    for i in range(np.shape(err)[0]):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.imshow(err[i,:,:])
        plt.colorbar()
        plt.savefig(outdir+name+'_crop_'+'x_'+str(i)+'.png')
        plt.close("all")


def performance_scatterplot(x_test,y_test,y_test_pred,outdir,name,nsamples=1):
    
    if y_test.size()[0]>1:
        # print(str(y_test.size()[0])+' samples for scatterplot')
        cubelength=y_test_pred.size()[-1]
        
        sq_y_test_pred=np.squeeze(y_test_pred.detach().numpy())
        sq_x_test=np.squeeze(x_test.detach().numpy())
        sq_y_test=np.squeeze(y_test.detach().numpy())
        
        labels=[]
        grainsizes=[]
        
        for sample in range(nsamples):
            for x in range(cubelength):
                for y in range(cubelength):
                    for z in range(cubelength):
                        if not (sq_x_test[sample,0,x,y,z] in labels):
                            labels.append(sq_x_test[sample,0,x,y,z])
                
    
        grain_avg_gt=[]
        grain_avg_pred=[]
        
        for label in labels:
            grainsize=0
            gt_grainstresses=[]
            pred_grainstresses=[]
            for sample in range(nsamples):
                for x in range(cubelength):
                    for y in range(cubelength):
                        for z in range(cubelength):
                            if sq_x_test[sample,0,x,y,z]==label:
                                gt_grainstresses.append(sq_y_test[sample,x,y,z])
                                pred_grainstresses.append(sq_y_test_pred[sample,x,y,z])
                                grainsize+=1
            grain_avg_gt.append(np.average(gt_grainstresses))
            grain_avg_pred.append(np.average(pred_grainstresses))
            grainsizes.append(grainsize)
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)


        for index,gt in enumerate(grain_avg_gt):
            ax1.scatter(gt, grain_avg_pred[index],alpha=(grainsizes[index]/max(grainsizes)))
    
        dummy_x = np.linspace(min(grain_avg_gt),max(grain_avg_gt),100)
        plt.plot(dummy_x, dummy_x, color='red',label="Ideally")
        plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        plt.xlabel('FFT ground truth (MPa)')
        plt.ylabel('NN prediction (MPa)')
        plt.legend(loc="upper left")
        fig.savefig(outdir+'latest_performance_'+name+'.png')
        plt.close("all")
        
    else:
        # print('1 sample for scatterplot')
        cubelength=y_test_pred.size()[-1]
        
        sq_y_test_pred=np.squeeze(y_test_pred.detach().numpy())
        sq_x_test=np.squeeze(x_test.detach().numpy())
        sq_y_test=np.squeeze(y_test.detach().numpy())
        
        labels=[]
        grainsizes=[]
        
        for x in range(cubelength):
            for y in range(cubelength):
                for z in range(cubelength):
                    if not (sq_x_test[0,x,y,z] in labels):
                        labels.append(sq_x_test[0,x,y,z])
                
    
        grain_avg_gt=[]
        grain_avg_pred=[]
        
        for label in labels:
            grainsize=0
            gt_grainstresses=[]
            pred_grainstresses=[]
            for x in range(cubelength):
                for y in range(cubelength):
                    for z in range(cubelength):
                        if sq_x_test[0,x,y,z]==label:
                            gt_grainstresses.append(sq_y_test[x,y,z])
                            pred_grainstresses.append(sq_y_test_pred[x,y,z])
                            grainsize+=1
            grain_avg_gt.append(np.average(gt_grainstresses))
            grain_avg_pred.append(np.average(pred_grainstresses))
            grainsizes.append(grainsize)
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        for index,gt in enumerate(grain_avg_gt):
            ax1.scatter(gt, grain_avg_pred[index],alpha=(grainsizes[index]/max(grainsizes)))
    
        dummy_x = np.linspace(min(grain_avg_gt),max(grain_avg_gt),100)
        plt.plot(dummy_x, dummy_x, color='red',label="Ideally")
        plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        plt.xlabel('FFT ground truth (MPa)')
        plt.ylabel('NN prediction (MPa)')
        plt.legend(loc="upper left")
        fig.savefig(outdir+'latest_performance_'+name+'.png')
        plt.close("all")

        return grain_avg_gt,grain_avg_pred,grainsizes


def performance_scatterplot_numpy(x_test,y_test,y_test_pred,outdir,name):
    
    cubelength=y_test_pred.shape[-1]
    
    sq_y_test_pred=np.squeeze(y_test_pred)
    sq_x_test=np.squeeze(x_test)
    sq_y_test=np.squeeze(y_test)
    
    labels=[]
    grainsizes=[]
    
    for x in range(cubelength):
        for y in range(cubelength):
            for z in range(cubelength):
                if not (sq_x_test[0,x,y,z] in labels):
                    labels.append(sq_x_test[0,x,y,z])
            

    grain_avg_gt=[]
    grain_avg_pred=[]
    
    for label in labels:
        grainsize=0
        gt_grainstresses=[]
        pred_grainstresses=[]
        for x in range(cubelength):
            for y in range(cubelength):
                for z in range(cubelength):
                    if sq_x_test[0,x,y,z]==label:
                        gt_grainstresses.append(sq_y_test[x,y,z])
                        pred_grainstresses.append(sq_y_test_pred[x,y,z])
                        grainsize+=1
        grain_avg_gt.append(np.average(gt_grainstresses))
        grain_avg_pred.append(np.average(pred_grainstresses))
        grainsizes.append(grainsize)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for index,gt in enumerate(grain_avg_gt):
        ax1.scatter(gt, grain_avg_pred[index],alpha=(grainsizes[index]/max(grainsizes)))

    dummy_x = np.linspace(min(grain_avg_gt),max(grain_avg_gt),100)
    plt.plot(dummy_x, dummy_x, color='red',label="Ideally")
    plt.xlabel("Ground truth")
    plt.ylabel("Prediction")
    plt.legend(loc="upper left")
    fig.savefig(outdir+'latest_performance_'+name+'.png')
    plt.close("all")


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

### FUNCTIONS BELOW FOR COMPARISON WITH SELF-CONSISTENT ANALYTICAL ESTIMATES

def rotateC_euler(C2unrot,euler):
    C4unrot=sec2fourth(C2unrot)
    # ROWENHORST
    c1=np.cos(euler[0])
    c=np.cos(euler[1])
    c2=np.cos(euler[2])
    s1=np.sin(euler[0])
    s=np.sin(euler[1])
    s2=np.sin(euler[2])
    g=np.array([[(c1*c2)-(s1*c*s2),(s1*c2)+(c1*c*s2),s*s2],
            [-(c1*s2)-(s1*c*c2),-(s1*s2)+(c1*c*c2),s*c2],
            [s1*s,-c1*s,c]
            ])
    # ROWENHORST
    
    ## ALTERNATIVE FORMULA: MY DEDUCTION
    c1=np.cos(euler[0])
    c2=np.cos(euler[1])
    c3=np.cos(euler[2])
    s1=np.sin(euler[0])
    s2=np.sin(euler[1])
    s3=np.sin(euler[2])

    g=np.array([[(c1*c3)-(s1*c2*s3),(c1*s3)+(s1*c2*c3),s1*s2],
            [-(s1*c3)-(c1*c2*s3),-(s1*s3)+(c1*c2*c3),c1*s2],
            [s2*s3,-(s2*c3),c2]
            ])
    ## ALTERNATIVE FORMULA: MY DEDUCTION
    ## BOTH FORMULAS ARE EQUIVALENT

    G=np.zeros(shape=[3,3,3,3,3,3,3,3])
    rotC=np.zeros(shape=[3,3,3,3])
    rotCvoigt=np.zeros(shape=[6,6])
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            for o in range(3):
                                for p in range(3):
                                    G[i,j,k,l,m,n,o,p]=\
                                    g[m,i]*g[n,j]*g[o,k]*g[p,l]
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    terms=0.0
                    for m in range(3):
                        for n in range(3):
                            for o in range(3):
                                for p in range(3):
                                    terms=terms+\
                                    (G[i,j,k,l,m,n,o,p]*C4unrot[m,n,o,p])
                    rotC[i,j,k,l]=terms
    
    rotCvoigt=fourth2sec(rotC)
    return rotCvoigt

def rotateC_quat(C2unrot,quat):
    C4unrot=sec2fourth(C2unrot)
    g=qu2om(quat)
    G=np.zeros(shape=[3,3,3,3,3,3,3,3])
    rotC=np.zeros(shape=[3,3,3,3])
    rotCvoigt=np.zeros(shape=[6,6])
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            for o in range(3):
                                for p in range(3):
                                    G[i,j,k,l,m,n,o,p]=\
                                    g[m,i]*g[n,j]*g[o,k]*g[p,l]
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    terms=0.0
                    for m in range(3):
                        for n in range(3):
                            for o in range(3):
                                for p in range(3):
                                    terms=terms+\
                                    (G[i,j,k,l,m,n,o,p]*C4unrot[m,n,o,p])
                    rotC[i,j,k,l]=terms
    
    rotCvoigt=fourth2sec(rotC)
    return rotCvoigt

def qu2eu_func(quat,P=-1):
    q0=quat[0]
    q1=quat[1]
    q2=quat[2]
    q3=quat[3]
    q03=(quat[0]**2)+(quat[3]**2)
    q12=(quat[1]**2)+(quat[2]**2)
    csi=(q03*q12)**0.5
    
    if csi==0:
        if q12==0:
            euler0=np.arctan((-2*P*q0*q3)/( (q0**2) - (q3 **2)))
            euler1=0
            euler2=0
        elif q03==0:
            euler0=np.arctan((2*q1*q2)/( (q1**2) - (q2 **2)))
            euler1=np.pi
            euler2=0
    else:
        euler0=np.arctan( ( ((q1*q3)-(P*q0*q2))/csi ) / ( ((-P*q0*q1)-(q2*q3))/csi ) )
        euler1=np.arctan((2*csi)/(q03-q12))
        euler2=np.arctan((((P*q0*q2)+(q1*q3))/csi)/(((q2*q3)-(P*q0*q1))/csi))
        
    return np.array([euler0,euler1,euler2],dtype='float64')

def getC_tial(s,angle):
    if (s=='tetra'):
        C11=183000
        C12=74000
        C22=C11
        C33=178000
        C13=C12
        C23=C12
        C44=105000
        C55=C44
        C66=78000
    
    elif (s=='cubic'):
        C11=183000
        C12=74000
        C22=C11
        C33=183000
        C13=C12
        C23=C12
        C44=105000
        C55=C44
        C66=105000

    Cunrot=np.eye(6)
    Cunrot=np.array([[C11,C12,C13,0,0,0],
       [C12,C22,C23,0,0,0],
       [C13,C23,C33,0,0,0],
       [0,0,0,C44,0,0],
       [0,0,0,0,C55,0],
       [0,0,0,0,0,C66]],dtype='float64')
    
    if len(angle)==3:
        Crot=rotateC_euler(Cunrot, angle)
    else:
        # angle=qu2eu_func(angle)
        # Crot=rotateC_euler(Cunrot,angle)
        ## USING NEWLY IMPLEMENTED FUNCTIONS (qu2om)
        Crot=rotateC_quat(Cunrot,angle)
    return Crot

def vr(C):
    kv=(2*C[0,0])/9 + (2*C[0,1])/9 + (4*C[0,2])/9 + C[2,2]/9
    
    kr=(- 2*(C[0,2]**2) + C[0,0]*C[2,2] + C[0,1]*C[2,2])/(C[0,0] + C[0,1] - 4*C[0,2] + 2*C[2,2])

    gveff= ((C[0,0]+C[0,1]-(4*C[0,2])+(2*C[2,2])))/6
    greff=(((C[0,0]+C[0,1])*C[2,2]-(2*(C[0,2]**2)))/6)/kv

    mu3= C[0,0]/2 - C[0,1]/2
    
    gv=(2*C[3,3])/5 + C[5,5]/5 + gveff/5 + mu3/5

    gr=1/(2/(5*C[3,3]) + 1/(5*C[5,5]) + 1/(5*greff) + 1/(5*mu3))

    ## DEBUG
    # if (abs((kv*greff)-(kr*gveff))>1e-10):
    #     print(kr)
    #     print(kv)
    #     print(greff)
    #     print(gveff)
    #     print('**Problem**');
    
    return [kv,kr,gv,gr,gveff,greff,mu3]

def pmw(C):
    du=0.01
    u=np.linspace(start=du,stop=1-du,num=int((1-du)/du))
    km=[]
    kM=[]
    gm=[]
    gM=[]
    
    for t1 in u:
        for t2 in u:
            [kmin,kmax,mumin,mumax]=pmw0(C,t1,t2)
            km.append(kmin)
            kM.append(kmax)
            gm.append(mumin)
            gM.append(mumax)
            
    kmin=min(km)
    kmax=max(kM)
    mumin=min(gm)
    mumax=max(gM)
    
    return [kmin,kmax,mumin,mumax]

def pmw0(C,t1,t2):
    
    [kv,kr,gv,gr,gveff,greff,mu3]=vr(C)
    
    gm=t1*min([C[3,3],mu3,greff,C[5,5]])
    gM=(1/t2)*max([C[3,3],mu3,gveff,C[5,5]])
    
    km=(kv*(gm - greff))/(gm - gveff)

    kM=(kv*(gM - greff))/(gM - gveff)

    am=-1/((4*gm)/3 + (kv*(gm - greff))/(gm - gveff))

    aM=-1/((4*gM)/3 + (kv*(gM - greff))/(gM - gveff))


    bm=(2*am)/15 - 1/(5*gm)

    bM=(2*aM)/15 - 1/(5*gM)

    delta=0

    Dm=2*bm*(gm - gveff) + am*(km - kv) + (am*bm*delta)/3 + 1

    DM=2*bM*(gM - gveff) + am*(kM - kv) + (aM*bM*delta)/3 + 1

    B2m=- (gm - mu3)/(5*(2*bm*(gm - mu3) + 1)) - (2*C[3,3] - 2*gm)/(5*(2*bm*(C[3,3] - gm) - 1)) - (gm - gveff)/(5*Dm) - (C[5,5] - gm)/(5*(2*bm*(C[5,5] - gm) - 1))

    B2M=- (gM - mu3)/(5*(2*bM*(gM - mu3) + 1)) - (2*C[3,3] - 2*gM)/(5*(2*bM*(C[3,3] - gM) - 1)) - (gM - gveff)/(5*DM) - (C[5,5] - gM)/(5*(2*bM*(C[5,5] - gM) - 1))

    kmin=km - (km - kv)/(2*bm*(gm - gveff) + 1)

    kmax=kM - (kM - kv)/(2*bM*(gM - gveff) + 1)

    mumin=gm + B2m/(2*B2m*bm + 1)

    mumax=gM + B2M/(2*B2M*bM + 1)
    
    return [kmin,kmax,mumin,mumax]

def sc0(ks,gs,C):

  [kv,kr,gv,gr,gveff,greff,mu3]=vr(C)

  a=-1/((4*gs)/3 + ks)

  xis=(gs*(8*gs + 9*ks))/(6*(2*gs + ks))

  newgs=5/(2/(C[3,3] + xis) + 1/(C[5,5] + xis) + 1/(mu3 + xis) + (a*(ks - kv) + 1)/(gveff + xis)) - xis

  newks=(gveff*kr + kv*xis)/(gveff + xis)
  
  return [newks,newgs]

def sc(ks0,gs0,C):

    err=99
    tol=1e-10

    while(err>tol):
        [ks,gs]=sc0(ks0,gs0,C)
        err=abs(ks-ks0)+abs(gs-gs0)
        ks0=ks
        gs0=gs
 
    return [ks,gs]

def getBerryman(C):

    # Voigt and Reuss bounds
    [kv,kr,gv,gr]=vr(C)[0:4]

    # Hashin & Shtrikman bounds
    [km,kM,gm,gM]=pmw(C)[0:4]

    # self-consistent
    [ks,gs]=sc(km,gm,C)[0:2]

    return [kr,ks,kv,km,kM,gr,gs,gv,gm,gM]

def isoStiff(K,G):
    
    lam=K-((2/3)*G)
    
    C11=(2*G)+lam
    C44=2*G
        
    C12=C11-C44
    
    C=np.array([[C11,C12,C12,0,0,0],
       [C12,C11,C12,0,0,0],
       [C12,C12,C11,0,0,0],
       [0,0,0,C44,0,0],
       [0,0,0,0,C44,0],
       [0,0,0,0,0,C44]])
    
    return C

def voigt2voigt(C):
        
    # from this notation
    #
    # [s11,s22,s33,s23,s13,s12]^t = C[e11,e22,e33,2e23,2e13,2e12]
    #
    # to the Voigt notation "with sqrt(2)" ie
    #
    # [s11,s22,s33,sqrt(2)s23,sqrt(2)s13,sqrt(2)s12]^t = C[e11,e22,e33,sqrt(2)e23,sqrt(2)e13,sqrt(2)e12]
    
    Cb=np.zeros((6,6),dtype="float64")
    Cb[0:3,0:3]=C[0:3,0:3]
    Cb[0:3,3:6]=C[0:3,3:6]*np.sqrt(2)
    Cb[3:6,0:3]=C[3:6,0:3]*np.sqrt(2)
    Cb[3:6,3:6]=C[3:6,3:6]*2
    
    return Cb

def voigt2voigtInv(C):

    Cb=np.zeros((6,6),dtype="float64")
    Cb[0:3,0:3]=C[0:3,0:3]
    Cb[0:3,3:6]=C[0:3,3:6]/np.sqrt(2)
    Cb[3:6,0:3]=C[3:6,0:3]/np.sqrt(2)
    Cb[3:6,3:6]=C[3:6,3:6]/2
    
    return Cb

def mean_field_stress(symmetry,angle,strain_load):
    
    # tenseur d'élasticité du grain i en notation '2'
    # Cref=getC('tetra',[0,0,0])
    Cref=getC_tial(symmetry,angle)

    # chargement imposé en déformation notation "2"
    # eps0=np.expand_dims(np.array([0,.5,0,0,0,0],dtype='float64'),axis=1)
    eps0=np.array(strain_load,dtype='float64')

    [kappaR,kappaSC,kappaV,kappaHSl,kappaHSu,muR,muSC,muV,muHSl,muHSu]=getBerryman(Cref)

    K0=kappaSC
    G0=muSC

    # notation sqrt(2)

    C0_sqrt=isoStiff(K0,G0)

    # notation 2

    C0=voigt2voigtInv(C0_sqrt)

    nu0=(3*K0-2*G0)/(2*(3*K0+G0))

    # Eq. 15

    Q11=16/15*G0/(1-nu0)
    Q22=Q11
    Q33=Q11

    Q12=2*G0/15*(1+5*nu0)/(1-nu0)
    Q23=Q12
    Q13=Q12

    Q44=(Q11-Q12)/2
    Q55=Q44
    Q66=Q44

    # convention avec "2esp12"

    Q=np.array([[Q11,Q12,Q13,0,0,0],
        [Q12,Q22,Q23,0,0,0],
        [Q13,Q23,Q33,0,0,0],
        [0,0,0,Q44,0,0],
        [0,0,0,0,Q55,0],
        [0,0,0,0,0,Q66]],dtype='float64')


    # teste=np.array([[0,1,2,3,4,5],
    #     [10,11,12,13,14,15],
    #     [20,21,22,23,24,25],
    #     [30,31,32,33,34,35],
    #     [40,41,42,43,44,45],
    #     [50,51,52,53,54,55]],dtype='float64')
       
    # notation avec "sqrt(2)"
       
    Q_sqrt=voigt2voigt(Q)
    C_sqrt=voigt2voigt(Cref)

    S_sqrt=np.linalg.inv(C_sqrt)
    S0_sqrt=np.linalg.inv(C0_sqrt)
    I=np.eye(6)

    Gam_sqrt=np.linalg.inv(I+(Q_sqrt*(S_sqrt-S0_sqrt)))

    # notation '2'
    # Gam=voigt2voigtInv(Gam_sqrt)

    # contrainte moyenne notation '2'
    sig0=np.einsum('ij,j->i',C0_sqrt,eps0)

    # moyenne dela contrainte dans le grain i
    # sigi=Gam*sig0
    sigi=np.einsum('ij,j->i',Gam_sqrt,sig0)
    
    return sigi

    # EXAMPLE
    # a=mean_field_stress('tetra', [0,0,0], [1,0,0,0,0,0])

def SC_plotter(SC_output_dir,x_dataset,y_fft,symmetry,sample,load,plotted_component,plotted_crop):
    ## SC_output_dir: where in outdir to plot the results
    ## x_dataset is the desired quaternion tensor
    ## y_fft is the corresponding fft ground truth
    ## symmetry is 'tetra'
    ## sample is just the index such as 0
    ## load is the applied epsilon in vector notation (eps_xx, eps_yy, eps_zz, eps_yz, eps_xz, eps_xy)
    ## plotted component is like "xx"
    ## plotted crop is like 0


    dict_idx_sc_estimates = {
    "xx":0,
    "yy":1,
    "zz":2,
    "yz":3,
    "xz":4,
    "xy":5
    }

    dict_idx_fft = {
    "xx":0,
    "xy":1,
    "xz":2,
    "yy":3,
    "yz":4,
    "zz":5
    }

    micro=np.squeeze(x_dataset[sample:sample+1])
    cube_size=micro.shape[1]
    quatlist=[]
    stresslist=[]

    for x in range(cube_size):
        for y in range(cube_size):
            for z in range(cube_size):
                if not arreq_in_list(micro[:,x,y,z], quatlist):
                    quatlist.append(micro[:,x,y,z])
                    stresslist.append(mean_field_stress(symmetry, micro[:,x,y,z], load))

    stresscube=np.zeros((6,cube_size,cube_size,cube_size),dtype='float64')

    for x in range(cube_size):
        for y in range(cube_size):
            for z in range(cube_size):
                for idx,quater in enumerate(quatlist):
                    if np.all(np.array(quater)==np.array(micro[:,x,y,z])):
                        stresscube[:,x,y,z]=stresslist[idx]
    
    stresscube=np.expand_dims(stresscube, axis=0)

    plotcropsvector('sc_sig_',stresscube,plotted_component,SC_output_dir+'/')
    performance_scatterplot_numpy(x_dataset[sample:sample+1],
                            y_fft[sample:sample+1,dict_idx_fft[plotted_component]:dict_idx_fft[plotted_component]+1,:,:,:],
                            stresscube[0:1,dict_idx_sc_estimates[plotted_component],:,:,:],
                            SC_output_dir+'/',
                            'comparison_sigma_sc_vs_fft_'+plotted_component)

######### END OF SC ESTIMATE FUNCTIONS
    
def vm_vector(sigma_fft_unibatch):
    term11=sigma_fft_unibatch[0,:,:,:]
    term12=sigma_fft_unibatch[1,:,:,:]
    term13=sigma_fft_unibatch[2,:,:,:]
    term22=sigma_fft_unibatch[3,:,:,:]
    term23=sigma_fft_unibatch[4,:,:,:]
    term33=sigma_fft_unibatch[5,:,:,:]

    sigma_vm_unibatch=np.sqrt(0.5*(np.square(term11-term22)+
                                   np.square(term22-term33)+
                                   np.square(term33-term11))
                                   +3*(np.square(term12)+
                                       np.square(term23)+
                                       np.square(term13)))
    
    return sigma_vm_unibatch
    # torch.Size([32, 32, 32])

def vm_matrix(sigma_nn_unibatch):

    term11=sigma_nn_unibatch[:,0,0,:,:,:]
    term12=sigma_nn_unibatch[:,0,1,:,:,:]
    term13=sigma_nn_unibatch[:,0,2,:,:,:]
    term22=sigma_nn_unibatch[:,1,1,:,:,:]
    term23=sigma_nn_unibatch[:,1,2,:,:,:]
    term33=sigma_nn_unibatch[:,2,2,:,:,:]

    sigma_vm_unibatch=np.sqrt(0.5*(np.square(term11-term22)+
                                   np.square(term22-term33)+
                                   np.square(term33-term11))
                                   +3*(np.square(term12)+
                                       np.square(term23)+
                                       np.square(term13)))
    
    return sigma_vm_unibatch
    # torch.Size([1, 32, 32, 32])

def loadb(filename='o',dL='o',prec='o'):
    
    # prec is int or double
    
    # define type
    if filename=='o' and dL=='o' and prec=='o':
        return print("Insert valid parameters")
    
    if filename!='o' and dL!='o' and prec=='o':
        prec=np.float64
    elif filename!='o' and dL=='o' and prec=='o':
        prec=np.float64
        dL=2

    if prec=='int':
        prec=np.int32
    
    fid=open(filename, 'rb')
    x=np.fromfile(fid, prec)
    count=len(x)
    
    # get dimension and lengths
    if filename!='o' and dL=='o' and prec=='o':
        #special case: no d, no L
        L2d=round(count^(1/2))
        L3d=round(count^(1/3))
        if L2d^2==count and L3d^3!=count:
            L=np.array([L2d,L2d])
        elif L3d^3==count and L2d^2!=count:
            L=np.array([L3d,L3d,L3d])
        else:
            L=np.array([L2d,L2d])
    elif np.size(np.array(dL))==1:
        d=dL
        L=round(count**(1./d))*np.ones((1,d))
    else:
        L=dL
    
    if np.prod(L)!=count:
        print('Warning: could not reshape acc. to specified dimension')
        print('L='+str(L)+'; count='+str(count)+'~= prod(L)='+str(np.prod(L))+' file: '+filename)
    
    elif max(np.shape(L))>1:
        x=np.reshape(x,tuple(map(tuple, L.astype(int)))[0])

    rotationcorrection=np.zeros(shape=np.shape(x),dtype=x.dtype)
        
    for index in range(np.shape(x)[0]):
        rotationcorrection[index,:,:]=np.transpose(x[:,:,index])
    
    return rotationcorrection

def dataset_generator(input_dir='./inputs',output_dir='./outputs',write_dir='./',cube_side=32,precision='float32',
                      file_name_str='S',ors_name_str='o',counting_index_starter=2001):
    
    if precision=='float32':
        prec=np.float32
    else:
        print('Precision not available, using np.float32\n')
        prec=np.float32
    
    nb_samples=int(0.5*(len([entry for entry in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, entry))])))

    quatdatabase=np.zeros(shape=(nb_samples,4,cube_side,cube_side,cube_side),dtype=prec)
    sigmadatabase=np.zeros(shape=(nb_samples,6,cube_side,cube_side,cube_side),dtype=prec)

    for file in os.listdir(os.fsencode(input_dir)):
        filename = os.fsdecode(file)
        if filename.endswith(ors_name_str):
            print(filename)
            # sampleindex=int(filename.replace("o","").replace("S",""))-1
            sampleindex=int(filename.replace(ors_name_str,"").replace(file_name_str,""))-counting_index_starter # culpa do Philipp
            print(sampleindex)
            
            micro=loadb(input_dir+'/'+filename.replace(ors_name_str,""),dL=3,prec='int')
            eulerslist = loadtxt(input_dir+'/'+filename)
            quatlist=np.zeros(shape=(eulerslist.shape[0],4),dtype=prec)
            sigma_xx=loadb(output_dir+'/'+filename.replace(ors_name_str,"")+'_Sigma_xx',dL=3,prec='double')
            sigma_xy=loadb(output_dir+'/'+filename.replace(ors_name_str,"")+'_Sigma_xy',dL=3,prec='double')
            sigma_xz=loadb(output_dir+'/'+filename.replace(ors_name_str,"")+'_Sigma_xz',dL=3,prec='double')
            sigma_yy=loadb(output_dir+'/'+filename.replace(ors_name_str,"")+'_Sigma_yy',dL=3,prec='double')
            sigma_yz=loadb(output_dir+'/'+filename.replace(ors_name_str,"")+'_Sigma_yz',dL=3,prec='double')
            sigma_zz=loadb(output_dir+'/'+filename.replace(ors_name_str,"")+'_Sigma_zz',dL=3,prec='double')
            
            sigmadatabase[sampleindex,0,:,:,:]=sigma_xx[:,:,:]
            sigmadatabase[sampleindex,1,:,:,:]=sigma_xy[:,:,:]
            sigmadatabase[sampleindex,2,:,:,:]=sigma_xz[:,:,:]
            sigmadatabase[sampleindex,3,:,:,:]=sigma_yy[:,:,:]
            sigmadatabase[sampleindex,4,:,:,:]=sigma_yz[:,:,:]
            sigmadatabase[sampleindex,5,:,:,:]=sigma_zz[:,:,:]
            
            for line in range(eulerslist.shape[0]):
                quatlist[line,:]=eu2quat(eulerslist[line,1:4])
                for index1 in range(cube_side):
                    for index2 in range(cube_side):
                        for index3 in range(cube_side):
                            # print(micro[index1,index2,index3])
                            if micro[index1,index2,index3]==line+1:
                                quatdatabase[sampleindex,:,index1,index2,index3]=quatlist[line,:]
                
    np.save(write_dir+'quat_'+str(nb_samples)+'_'+str(cube_side)+'^3.npy',quatdatabase)
    np.save(write_dir+'sigma_'+str(nb_samples)+'_'+str(cube_side)+'^3.npy',sigmadatabase)

class error_data:
    def __init__(self, filename, path):
        self.filename = filename
        self.path = path

    def load(self):
        self.data = 100 * np.loadtxt(self.path + self.filename, dtype=float)

    def pdf(self, kernel="gaussian", points=1000):
        self.kde = KernelDensity(kernel=kernel)
        self.kde.fit(self.data[:, None])

        # Generate points for the KDE plot
        self.x = np.linspace(min(self.data), max(self.data), points)
        self.log_density = self.kde.score_samples(self.x[:, None])

        # Calculate cumulative distribution function (CDF)
        self.pdf = np.exp(self.log_density)

    def cdf(self):
        self.cdf = np.cumsum(self.pdf) * (
            self.x[1] - self.x[0]
        )  # Numerical integration