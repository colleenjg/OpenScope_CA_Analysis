# OpenScope Credit Assignment Project Code (minimal version)

## 1. Description
This repository contains the code for analyzing the data from the Credit Assignment project, an [**Allen Institute for Brain Science**](https://alleninstitute.org/what-we-do/brain-science/) [**OpenScope**](https://alleninstitute.org/division/mindscope/openscope/) project. 

The dataset is described and characterized in [Gillon, Lecoq _et al._, 2023, _Sci Data_](https://doi.org/10.1038/s41597-023-02214-y). Analyses and results are published in [Gillon, Pina _et al._, 2024, _J Neurosci_](https://www.jneurosci.org/content/44/5/e1009232023).

This branch provides a **minimal** example, with minimal dependencies. It can be used to rerun dataset-focused analyses (different from those in [Gillon, Pina, Lecoq _et al._, 2021, _bioRxiv_](https://www.biorxiv.org/content/10.1101/2021.01.15.426915) or [Gillon, Pina _et al._, 2024, _J Neurosci_](https://www.jneurosci.org/content/44/5/e1009232023).

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
* `run_paper_figures.py`: run, analyse and plot paper figures
* **`analysis`**: analysis scripts, including the Session and Stim objects
* **`plot_fcts`**: plotting scripts
* **`paper_fig_util`**: scripts to organize and generate the paper figures
* **`util`**: utilities module

## 5. Data
The full dataset for this project is hosted [here](https://gui.dandiarchive.org/#/dandiset/000037) in the DANDI archive in [NWB](https://www.nwb.org/) format. The associated metadata can be found [here](https://github.com/jeromelecoq/allen_openscope_metadata/tree/master/projects/credit_assignement). The subset of data used in the paper (33 sessions, ~15 GB total) can be downloaded by running, from the main directory of the repository:  
`python util/download_util.py --output path/to/save/`

Code to generate the stimuli used in these experiments can be found [here](https://github.com/colleenjg/cred_assign_stimuli).  

## 6. Figure example notebooks

The following notebooks give examples of how to download the data, and run the analyses and produce the figure panels for [Gillon, Lecoq _et al._, 2023](https://doi.org/10.1038/s41597-023-02214-y).

| Run in Binder | Run the Google Colab notebook |
| ------------- | ----------------------------- |
| [![Run in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/colleenjg/OpenScope_CA_Analysis/minimal?labpath=run_paper_figures.ipynb) | [![Run the Google Colab notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/colleenjg/OpenScope_CA_Analysis/blob/minimal/run_paper_figures_colab.ipynb) |

The contents of the **Binder** and **Google Colab** notebooks differ somewhat, due to the resources available: 
* **Binder:** conda env. is installed automatically (+), but this can be slow, and only limited compute resources are available (-).  
* **Google Colab:** less environment control (-), but more substantial compute resources are available (+).  


## 7. Authors
This code was written by:

* Colleen Gillon (colleen _dot_ gillon _at_ mail _dot_ utoronto _dot_ ca)
* Jay Pina, Joel Zylberberg, and Blake Richards

Please do not hesitate to contact the authors or open an issue/pull request, if you have trouble using the data or the codebase or improvements to propose.  


## 8. Citations

To cite the **dataset** paper:
```
@Article{GillonLecoq2023,
  title={Responses of pyramidal cell somata and apical dendrites in mouse visual cortex over multiple days},
  author={Gillon, Colleen J. and Lecoq, J{\'e}r{\^o}me A. and Pina, Jason E. and Ahmed, Ruweida and Billeh, Yazan and Caldejon, Shiella and Groblewski, Peter and Henley, Timothy M. and Kato, India and Lee, Eric and Luviano, Jennifer and Mace, Kyla and Nayan, Chelsea and Nguyen, Thuyanh and North, Kat and Perkins, Jed and Seid, Sam and Valley, Matthew T. and Williford, Ali and Bengio, Yoshua and Lillicrap, Timothy P. and Zylberberg, Joel and Richards, Blake A.},
  journal={Scientific Data},
  year={2023},
  date={May 2023},
  publisher={Cold Spring Harbor Laboratory},
  volume={10},
  number={1},
  pages={287},
  issn={2052-4463},
  doi={10.1038/s41597-023-02214-y},
  url={https://www.nature.com/articles/s41597-023-02214-y},
}
```

To cite the **analysis** paper:
```
@Article{GillonPina2024,
  title={Responses to pattern-violating visual stimuli evolve differently over days in somata and distal apical dendrites},
  author={Gillon, Colleen J. and Pina, Jason E. and Lecoq, J{\'e}r{\^o}me A. and Ahmed, Ruweida and Billeh, Yazan and Caldejon, Shiella and Groblewski, Peter and Henley, Timothy M. and Kato, India and Lee, Eric and Luviano, Jennifer and Mace, Kyla and Nayan, Chelsea and Nguyen, Thuyanh and North, Kat and Perkins, Jed and Seid, Sam and Valley, Matthew T. and Williford, Ali and Bengio, Yoshua and Lillicrap, Timothy P. and Richards, Blake A. and Zylberberg, Joel},
  journal={Journal of Neuroscience},
  year = {2024},
  date = {Jan 2024},
  publisher = {Society for Neuroscience},
  volume = {44},
  number = {5},
  pages = {1-22},
  issn = {0270-6474},
  doi = {10.1523/JNEUROSCI.1009-23.2023},
  url = {https://www.jneurosci.org/content/44/5/e1009232023},
}
```