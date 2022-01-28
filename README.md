# Credit Assignment Project Code

## 1. Description
This repository contains the code for analyzing the Credit Assignment project, an [**Allen Institute for Brain Science**](https://alleninstitute.org/what-we-do/brain-science/) [**OpenScope**](https://alleninstitute.org/what-we-do/brain-science/news-press/press-releases/openscope-first-shared-observatory-neuroscience) project. 

The experiment details, analyses and results are published in [Gillon _et al._, 2021, _bioRxiv_](https://www.biorxiv.org/content/10.1101/2021.01.15.426915v2).

## 2. Installation
To run the code, you should install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://conda.io/miniconda.html).

Once these are installed, you can simply use the appropriate `.yml` file to create a conda environment. For example, if using Ubuntu or Mac OS, open a terminal, go to the repository directory, and enter:

1. `conda env create -f osca.yml`  
2. `source activate osca`  

The code is written in `Python 3`. 

## 3. Use
Once installed, when using the codebase, simply activate the environment:

`source activate osca`

All of the appropriate libraries should then be loaded, and the modules can be imported for use in ipython, python scripts, or jupyter notebooks, for example.

## 4. Scripts and modules
* `run_paper_figures.py`: run, analyse and plot paper figures (for example usage, see the `paper_figures` folder)   
* **`analysis`**: analysis scripts, including the Session and Stim objects
* **`sess_util`**: session specific utilities module
* **`plot_fcts`**: plotting scripts
* **`paper_fig_util`**: scripts to organize and generate the paper figures
* **`examples`**: example notebook for using the Session and Stim objects 

## 5. Data
The full dataset for this project is hosted [here](https://gui.dandiarchive.org/#/dandiset/000037) in the DANDI archive in [NWB](https://www.nwb.org/) format. The associated metadata can be found [here](https://github.com/jeromelecoq/allen_openscope_metadata/tree/master/projects/credit_assignement). The subset of data used in the paper (33 sessions, ~15 GB total) can be downloaded by running, from the main directory of the repository:  
`python sess_util/sess_download_util.py --output path/to/save/`

Code to generate the stimuli used in these experiments can be found [here](https://github.com/colleenjg/cred_assign_stimuli).  

## 6. Example notebooks

The following notebooks give examples of how to download the data, and run the paper analyses.

| Run in Binder | View the notebook | Run the Google Colab notebook |
| ------------- | -------------------- | ------------------- |
| [![Run in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/colleenjg/OpenScope_CA_Analysis/main?labpath=run_paper_figures.ipynb) | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/colleenjg/OpenScope_CA_Analysis/blob/main/run_paper_figures.ipynb?flush_cache=true) | [![Run the Google Colab notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/colleenjg/OpenScope_CA_Analysis/blob/main/run_paper_figures_colab.ipynb) |

**Binder:** conda env. is already installed (+), but only limited compute resources are available (-).  
**Google Colab:** conda env. must first be installed (-), but more substantial compute resources are available (+).  


## 7. Authors
This code was written by:

* Colleen Gillon (colleen _dot_ gillon _at_ mail _dot_ utoronto _dot_ ca)
* Jay Pina, Joel Zylberberg, and Blake Richards

Please do not hesitate to contact the authors if you have trouble using the data or the codebase.  

**Note:** The module `Dataset2p.py` under `sess_util` contains code shared by authors at the Allen Institute for Brain Science, The authors of the code cannot guarantee support for its usage.

