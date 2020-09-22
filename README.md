# Allen Institute for Brain Science OpenScope Program - Credit Assignment Project Code

## 1. Description
This is the code for analyzing the Credit Assignment Project data collected as part of the [**Allen Institute for Brain Science (AIBS) OpenScope**](https://alleninstitute.org/what-we-do/brain-science/) project.

## 2. Installation
The code itself can be obtained by cloning the [AIBS\_Analysis Github repo](https://github.com/colleenjg/AIBS_Analysis.git).

However, to run the code, you should install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://conda.io/miniconda.html), as well as [pip](https://pip.pypa.io/en/stable/).

Once these are installed, you can simply use the appropriate `.yml` 
file to create a conda environment. For example, if using Ubuntu or Mac OS, open a terminal, go to the repo directory, and enter:

`conda env create -f aibs3.yml`

Conda should then handle installing all the necessary dependencies for you.

The code is written in `Python 3`. 

## 3. Use
Once installed, you simply activate the environment:

`source activate aibs3`

All of the appropriate libraries should then be loaded, and the modules can be imported for use in ipython, python scripts, or jupyter notebooks, for example.

**Scripts and modules**:

* `run_acr_sess_analysis.py`: run and plot specific analyses across sessions
* `run_roi_analysis.py`: run and plot specific analyses on ROI data
* `run_running_analysis.py`: run and plot specific analyses on running data
* `run_pupil_analysis.py`: run and plot specific analyses on pupil data
* `run_logreg.py`: run, analyse and plot logistic regressions on the ROI data
* **`analysis`**: Session object as well as session data analysis module
* **`plot_fcts`**: module with functions to plot analysis results from saved dictionaries or dataframes 
* **`sess_util`**: session specific utilities module
* **`examples`**: example notebook for using Session/Stim objects 
* **`scripts_under_dev`**: analysis scripts under development
* **`prev_scripts`**: analysis scripts not currently maintained

## 4. Authors
This code was written by:

* Colleen Gillon  (colleen.gillon@mail.utoronto.ca)
* Jay Pina (jaypina@yorku.ca)
* Joel Zylberberg (joelzy@yorku.ca)
* Blake Richards  (blake.richards@mcgill.ca)

The module `sess_util.Dataset2p.py` contains code shared by authors at the AIBS. The authors of the code cannot guarantee support for its usage.
