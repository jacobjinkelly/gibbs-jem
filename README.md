# Gibbs-JEM

Code for the paper:

> Jacob Kelly, Richard Zemel, Will Grathwohl. "Directly Training Joint Energy-Based Models for Conditional Synthesis and Calibrated Prediction of Multi-Attribute Data." _ICML UDL_ (2021).
> [[arxiv]](#) [[bibtex]](#bibtex)

# Environment

Create a conda environment with

```
conda env create -f environment.yml
```

# Data

## UTZappos

Create the directory for the dataset

```
mkdir -p data/utzappos
```

Download the [UTZappos dataset](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/) at this [link](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images-square.zip). 
Unzip and place in `data/utzappos`. The path should be `data/utzappos/ut-zap50k-images-square`.

Download the metadata from [here](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-data.zip). 
Unzip, and place the file `meta-data.csv` at the path `data/utzappos/ut-zap50k-images-square/meta-data.csv`.

## CelebA

Create the directory for the dataset

```
mkdir -p data/celeba
```

Download the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) at this [link][https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ]. Unzip and place in `data/celeba`. The path should be `data/celeba/img_align_celeba`.

Download the metadata [here](https://drive.google.com/file/d/0B7EVK8r0v71pblRyaVFSWGxPY0U/view?usp=sharing&resourcekey=0-YW2qIuRcWHy_1C2VaRGL3Q). 
Place `list_attr_celeba.txt` in `data/celeba/list_attr_celeba.txt`.

## BibTeX

```
@inproceedings{kelly2021gibbsjem,
  title={Directly Training Joint Energy-Based Models for Conditional Synthesis and Calibrated Prediction of Multi-Attribute Data},
  author={Kelly, Jacob and Zemel, Richard and Grathwohl, Will},
  journal={arXiv preprint arXiv:2107.#},
  year={2021}
}
```
