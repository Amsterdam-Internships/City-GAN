# CityGAN: Automatic addition and removal of objects

This repo contains the code for my MSc AI thesis project, exploring the use of GANs for the Municipality of Amsterdam. More specifically, I tried to reproduce and improve the CopyPasteGAN ([Arandejovic, 2018](https://arxiv.org/abs/1905.11369)) for object discovery. The ultimate goal is to find objects of a specific object class automatically in an image, and be able to "cut" that object from the source image, and place it realistically in a target image. 
The *_How it works_* section below contains more technical information about the model and reasoning.

![](media/examples/emojis.png)

---


## Project Folder Structure

The project setup is inspired by the [Pix2Pix framework](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), and is structured as follows:

1) [`models`](./models): Folder containing all model classes
2) [`data`](./data): Folder containing all dataset classes and helper functions
3) [`jobscripts`](./jobscripts): Folder containing all bash scripts used for generating datasets and training models on a GPU-cluster ([https://userinfo.surfsara.nl/](Surfsara))
4) [`options`](./options): Folder containing all command line options for various phases in training. Additional options can be defined in the model class definitions. 
5) [`util`](./util): Folder containing all utlity functions, including the visualizer (Visdom)


---


## Installation

Explain how to set up everything. 
Let people know if there are weird dependencies - if so feel free to add links to guides and tutorials.

1) Clone this repository:
    ```bash
    git clone https://github.com/Amsterdam-Internships/City-GAN
    ```

2) Install all dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3) Create a ./datasets directory with following structure:

---


## Usage

Explain example usage, possible arguments, etc. E.g.:

To train... 


```
$ python train.py --some-importang-argument
```

---


## How it works

Explain roughly how the code works, what the main components are, how certain crucial steps are performed...

---
## Acknowledgements


Don't forget to acknowledge any work by others that you have used for your project. Add links and check whether the authors have explicitly stated citation preference for using the DOI or citing a paper or so. 
For example:

Our code uses [YOLOv5](https://github.com/ultralytics/yolov5) [![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)

