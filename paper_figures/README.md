# Paper figures

This folder contains all of the data panels in the paper, organized by figure. 
Each panel is accompanied by a `json` file that contains its source data, and 
from which the panel can be directly replotted.

**Example usage:**  
- To generate Fig. 2C **from start to finish** (analysis + plotting), 
run, from the main directory:   
`python run_paper_figures.py --figure 2 --panel C --full_power --overwrite --parallel --datadir /path/to/data`  
- To instead **only replot** Fig. 2C from its source data, run from the main directory:  
`python run_paper_figures.py --figure 2 --panel C --full_power --plot_only`  

**Notes:** 
- `--full_power`: analysis is run with **full statistical power**, as was done 
for the paper (e.g., using 1e4 permutations, for permutation test). If 
running the full analysis, this means that more time and resources will be 
needed to run the code. If omitted, a lower power version of the panel will be  
generated instead, if applicable, and saved under a modified name, 
e.g. `Fig2/panel_C_lower_power.svg`.
- `--plot_only`: panel is replotted from the source data, if it exists. The 
full analysis is not run.
- `--overwrite`: if the source data for the panel exists, it is overwritten. 
If used with `--plot_only` as well, only the existing panel plot is overwritten.
- `--parallel`: analysis is run using all available CPU cores, to increase code 
efficiency, and reduce execution time.
- `--datadir`: path to the full dataset (e.g., in NWB format).  
- **Details on additional arguments** can be found in `run_paper_figures.py` or 
can be printed to the console by running `python run_paper_figures.py --help`, 
from the main directory.

