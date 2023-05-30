# OpenScope Credit Assignment Project Code

## 1. Description
This repository contains the code for analyzing the data from the Credit Assignment project, an [**Allen Institute for Brain Science**](https://alleninstitute.org/what-we-do/brain-science/) [**OpenScope**](https://alleninstitute.org/division/mindscope/openscope/) project. 

The dataset is described and characterized in [Gillon _et al._, 2023](https://doi.org/10.1038/s41597-023-02214-y). Analyses and results are published in [Gillon _et al._, 2021, _bioRxiv_](https://www.biorxiv.org/content/10.1101/2021.01.15.426915).

## 2. Installation
To run the code, you can install a conda-based environment manager (e.g., [Anaconda](https://www.anaconda.com/), [Miniconda](https://conda.io/miniconda.html) or [Mamba](https://mamba.readthedocs.io/en/latest/installation.html)).

Once the conda-based environment manager is installed, use `osca.yml` to create the `osca` environment with all the packages needed to run this code. For example, if using Ubuntu or Mac OS, open a terminal, go to the repository directory, and run:

`conda env create -f osca.yml`  

Alternatively, if you prefer to use a different environment manager, install required packages specified in `requirements.txt`. 

This code is written in `Python 3`, and has been tested with `Python 3.8`.

## 3. Use
Once installed, when using the codebase, simply activate the environment:

`source activate osca`  

All of the appropriate libraries should then be loaded, allowing the scripts and notebooks provided in the repo to be run.

## 4. Scripts and modules
* `run_paper_figures.py`: run, analyse and plot paper figures (for example usage, see the **`paper_figures`** folder)
* **`analysis`**: analysis scripts, including the Session and Stim objects
* **`sess_util`**: session specific utilities module
* **`plot_fcts`**: plotting scripts
* **`paper_fig_util`**: scripts to organize and generate the paper figures
* **`examples`**: example notebook for using the Session and Stim objects 

## 5. Data
The full dataset for this project is hosted [here](https://gui.dandiarchive.org/#/dandiset/000037) in the DANDI archive in [NWB](https://www.nwb.org/) format. The associated metadata can be found [here](https://github.com/jeromelecoq/allen_openscope_metadata/tree/master/projects/credit_assignement). The subset of data used in the paper (33 sessions, ~15 GB total) can be downloaded by running, from the main directory of the repository:  
`python sess_util/sess_download_util.py --output path/to/save/`

Code to generate the stimuli used in these experiments can be found [here](https://github.com/colleenjg/cred_assign_stimuli).  

## 6. Figure example notebooks

The following notebooks give examples of how to download the data, and run the paper analyses. Note that the organization of figures and panels has been updated since [Gillon _et al._, 2021, _bioRxiv_](https://www.biorxiv.org/content/10.1101/2021.01.15.426915) was published.

| Run in Binder | Run the Google Colab notebook |
| ------------- | ----------------------------- |
| [![Run in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/colleenjg/OpenScope_CA_Analysis/main?labpath=run_paper_figures.ipynb) | [![Run the Google Colab notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/colleenjg/OpenScope_CA_Analysis/blob/main/run_paper_figures_colab.ipynb) |

The contents of the **Binder** and **Google Colab** notebooks differ somewhat, due to the resources available: 
* **Binder:** conda env. is installed automatically (+), but this can be slow, and only limited compute resources are available (-).  
* **Google Colab:** less environment control (-), but more substantial compute resources are available (+).  

## 7. Terminological notes
The following terms used in the codebase are considered equivalent to the corresponding terms used in the papers:
- "expected": equivalent to "consistent" and "pattern-matching" for both stimuli, and "uniform flow" for the visual flow stimulus specifically. 
- "unexpected": equivalent to "inconsistent" and "pattern-violating" for both stimuli, and "counter-flow" for the visual flow stimulus specifically.

## 8. Authors
This code was written by:

* Colleen Gillon (colleen _dot_ gillon _at_ mail _dot_ utoronto _dot_ ca)
* Jay Pina, Joel Zylberberg, and Blake Richards

Please do not hesitate to contact the authors or open an issue/pull request, if you have trouble using the data or the codebase or improvements to propose.  


