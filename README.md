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

### Splits

Splits for the data are taken from [[1]](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly). Download this [file](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip) and extract the directory, which should be called `xlsa17`.

### CUB

Download the [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset [here](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz). Extract the contents (it should be called `CUB_200_2011`) and move the folder to `data/`.  Get the class splits defined in [[1]](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly) with

```markdown
cp path/where/extracted/xlsa17/data/CUB/*txt path/of/repo/data/CUB_200_2011/CUB_200_2011/
```

Different splits are implemented by creating aliases. To create these aliases run

```markdown
$ pwd
path/of/repo
$ cd utils
$ python cub.py
```

## BibTeX

```
@inproceedings{kelly2021gibbsjem,
  title={Directly Training Joint Energy-Based Models for Conditional Synthesis and Calibrated Prediction of Multi-Attribute Data},
  author={Kelly, Jacob and Zemel, Richard and Grathwohl, Will},
  journal={arXiv preprint arXiv:2107.#},
  year={2021}
}
```
