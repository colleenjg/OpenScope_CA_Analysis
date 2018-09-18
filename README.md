# Allen Institute for Brain Science OpenScope Program - Credit Assignment Project Code

## 1. Description
This is the code for analyzing the Credit Assignment Project data collected as part
of the first Allen Institute for Brain Science OpenScope program run.

## 2. Installation
The code itself can be obtained by cloning the github repo at:

https://github.com/joelzy/AIBS_Analysis.git

However, to run the code, you should install Anaconda or Miniconda:

* https://www.anaconda.com/
* https://conda.io/miniconda.html

As well as pip:

* https://pip.pypa.io/en/stable/

Once these are installed, you can simply use the appropriate appropriate .yml 
file to create a conda environment. For example, if using a Mac, open a 
terminal, go to the repo directory, and enter:

* conda env create -f aibs_mac_env.yml

Conda should then handle installing all the necessary dependencies for you.

## 3. Use
Once installed, you simply activate the appropriate environment, e.g. for Mac:

* source activate aibs-mac

All of the appropriate libraries should then be loaded, and the modules can
be imported for use in ipython, python scripts, or jupyter notebooks.

## 4. Authors
This code was written by:

* Colleen Gillon  (colleen.gillon@mail.utoronto.ca)
* Joel Zylberberg (joel.zylberberg@gmail.com)
* Blake Richards  (blake.richards@utoronto.ca)

The code also uses some code snippets from authors at the Allen. The authors
of the code cannot guarantee support for its usage.
