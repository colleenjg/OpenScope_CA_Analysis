# Allen Institute for Brain Science OpenScope Program - Credit Assignment Project Code

## 1. Description
This is the code for analyzing the Credit Assignment Project data collected as part
of the first Allen Institute for Brain Science OpenScope program run.

## 2. Installation
The code itself can be obtained by cloning the github repo at:

https://github.com/colleenjg/AIBS_Analysis.git

However, to run the code, you should install Anaconda or Miniconda:

* https://www.anaconda.com/
* https://conda.io/miniconda.html

As well as pip:

* https://pip.pypa.io/en/stable/

Once these are installed, you can simply use the appropriate .yml 
file to create a conda environment. For example, if using Ubuntu or Mac OS, open 
a terminal, go to the repo directory, and enter:

* conda env create -f aibs3.yml

Conda should then handle installing all the necessary dependencies for you.

The code is written in Python 3. 

## 3. Use
Once installed, you simply activate the environment:

* source activate aibs3

All of the appropriate libraries should then be loaded, and the modules can
be imported for use in ipython, python scripts, or jupyter notebooks.

**Scripts and modules**:
* run\_acr\_sess_analysis.py: run and plot specific analyses across sessions
* run\_roi_analysis.py: run and plot specific analyses on ROI data
* run\_running_analysis.py: run and plot specific analyses on running data
* run\_pupil_analysis.py: run and plot specific analyses on pupil data
* run_logreg.py: run, analyse and plot logistic regressions on the ROI data
* **analysis**: Session object as well as session data analysis module
* **plot_fcts**: module with functions to plot analysis results from saved dictionaries or dataframes 
* **sess_util**: session specific utilities module
* **examples**: example notebook for using Session/Stim objects 
* **scripts_under_dev**: analysis scripts under development
* **prev_scripts**: iPython notebooks (not currently maintained)

## 4. Authors
This code was written by:

* Colleen Gillon  (colleen.gillon@mail.utoronto.ca)
* Jay Pina (jay.pina@pitt.edu)
* Joel Zylberberg (joel.zylberberg@gmail.com)
* Blake Richards  (blake.richards@utoronto.ca)

The code also uses some modules shared by authors at the Allen. The authors
of the code cannot guarantee support for its usage.
