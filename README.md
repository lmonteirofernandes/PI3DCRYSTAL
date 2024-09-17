<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">


<h3 align="center">Physics-Informed 3D Surrogate for Elastic Fields in Polycrystals</h3>

  <p align="center">
    This repository contains a physics-informed pipeline for surrogate modeling linear elasticity in 3D polycrystalline materials. Once trained, these translation-equivariant models can be exploited to solve ill-posed inverse problems in polycrystalline materials engineering such as crystallographic texture optimization.
    <br />
    <br />
  </p>
</div>


### Citation

Our paper is available on . Please cite our work with
```sh
@misc{ ,
	title =  ,
	author = ,
	month =  ,
	year =  }
  ```

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With
Our released implementation is tested on:
* Ubuntu 20.04
* Python 3.8.18
* PyTorch 2.0.1
* NVIDIA CUDA 11.7


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting started

### Prerequisites

* Clone the repository
* Create and launch a conda environment with
  ```sh
  conda create -n PI3DCRYSTAL python=3.8.18
  conda activate PI3DCRYSTAL
  ```
<!--### Installation-->
* Install dependencies
    ```sh
  pip install -r requirements.txt
  ```
  Note: for Pytorch CUDA installation follow https://pytorch.org/get-started/locally/.
  
<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Datasets and trained models
The datasets and trained models used in our experiments can be downloaded [here](https://cloud.minesparis.psl.eu/index.php/s/e1r25deh2MhlOpi) and should be placed inside the data and trained_models folders respectively.

### Training
To train a model you can use the `train.py` script provided.

### Testing 
The trained models can be tested with the `post_proc.ipynb` notebook.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

If you have any idea that would be useful to implement, please fork the repository and create a pull request. 

1. Fork the repository
2. Create a branch (`git checkout -b feature/my_feature`)
3. Commit your changes (`git commit -m 'Add my_feature'`)
4. Push to the branch (`git push origin feature/my_feature`)
5. Open a pull request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Lucas Monteiro Fernandes - https://www.linkedin.com/in/lucas-monteiro-fernandes-96b621171 - lucas.monteiro_fernandes@minesparis.psl.eu - lucasmon10@hotmail.com

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This project has received funding from the French Agence Nationale de la Recherche (ANR, ANR-21-FAI1-0003) and the Bundesministerium für Bildung und Forschung (BMBF, 01IS21091) within the French-German research project SMILE.

<p align="right">(<a href="#top">back to top</a>)</p>
