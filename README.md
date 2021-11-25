# Credit Assignment Project Code

## 1. Description
This repository contains the code for analyzing the Credit Assignment project, an [**Allen Institute for Brain Science**](https://alleninstitute.org/what-we-do/brain-science/) [**OpenScope**](https://alleninstitute.org/what-we-do/brain-science/news-press/press-releases/openscope-first-shared-observatory-neuroscience) project. 

The experiment details, analyses and results are published in [Gillon _et al._, 2021, _bioRxiv_](https://www.biorxiv.org/content/10.1101/2021.01.15.426915v2).

## 2. Installation
To run the code, you should install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://conda.io/miniconda.html), as well as [pip](https://pip.pypa.io/en/stable/).

Once these are installed, you can simply use the appropriate `.yml` 
file to create a conda environment. For example, if using Ubuntu or Mac OS, open a terminal, go to the repo directory, and enter:

1. `conda env create -f osca.yml`  
2. `source activate osca`  
3. `pip install pandas==1.1.5` # ignore the warning about a version conflict with the `allensdk` package.

The code is written in `Python 3`. 

## 3. Use
Once installed, when using the codebase, simply activate the environment:

`source activate osca`

All of the appropriate libraries should then be loaded, and the modules can be imported for use in ipython, python scripts, or jupyter notebooks, for example.

## 4. Scripts and modules
* `run_paper_figures.py`: run, analyse and plot paper figures _(UNDER DEVELOPMENT)_  

* **`main_scripts`**: main scripts from which to run and plot specific analyses...
    * `run_acr_sess_analysis.py`: ... across sessions
    * `run_roi_analysis.py`: ... on ROI data
    * `run_running_analysis.py`: ... on running data
    * `run_pupil_analysis.py`: ... on pupil data
    * `run_logreg.py`: ... using logistic regressions on the ROI data
* **`analysis`**: Session object as well as session data analysis module
* **`plot_fcts`**: module with functions to plot analysis results from saved dictionaries or dataframes 
* **`sess_util`**: session specific utilities module
* **`paper_fig_util`**: analysis scripts to generate the paper scripts
* **`examples`**: example notebook for using Session/Stim objects 
* **`scripts_under_dev`**: analysis scripts under development

## 5. Data
The data for this project is hosted [here](https://gui.dandiarchive.org/#/dandiset/000037) in the DANDI archive in [NWB](https://www.nwb.org/) format. The associated metadata can be found [here](https://github.com/jeromelecoq/allen_openscope_metadata/tree/master/projects/credit_assignement).   
**PLEASE NOTE:** We are currently working to update the codebase to interface with the NWB format.
&nbsp;

Code to generate the stimuli can be found [here](https://github.com/colleenjg/cred_assign_stimuli). 


## 6. Authors
This code was written by:

* Colleen Gillon (colleen.gillon@mail.utoronto.ca)
* Jay Pina (jaypina@yorku.ca)
* Joel Zylberberg (joelzy@yorku.ca)
* Blake Richards (blake.richards@mcgill.ca)

The module `sess_util.Dataset2p.py` contains code shared by authors at the Allen Institute for Brain Science. The authors of the code cannot guarantee support for its usage.
