# Paper figures

This folder contains all of the data panels in the paper, organized by figure. Each panel is accompanied by a `json` file that contains its source data, and from which the panel can be directly replotted. Figures for the **dataset descriptor paper** [Gillon _et al._, 2023](https://doi.org/10.1038/s41597-023-02214-y) are stored under `dataset_paper`. Other figures are for the **analysis paper** [Gillon _et al._, 2021, _bioRxiv_](https://www.biorxiv.org/content/10.1101/2021.01.15.426915).

**Example usage:**  
- To generate Fig. 3C **from start to finish** (analysis + plotting), 
run, from the main directory:   
`python run_paper_figures.py --figure 3 --panel C --full_power --overwrite --parallel --datadir path/to/data`  
- To instead **only replot** Fig. 3C from its source data, run from the main directory:  
`python run_paper_figures.py --figure 3 --panel C --full_power --plot_only`  
- To create figures, e.g., Fig. 6, from the **dataset paper** (under development), run from the main directory:  
`python run_paper_figures.py --paper dataset --figure 6`  


**Notes:** 
- `--full_power`: analysis is run with **full statistical power**, as was done for the paper (e.g., using 1e4 permutations, for permutation tests). If running the full analysis, this means that more time and resources will be needed to run the code.<sup>[1](#1)</sup> If omitted, a lower power version of the panel will be generated instead, if applicable, and saved under a modified name, e.g. `analysis_paper/Fig3/panels_with_diff_params/panel_C_lower_power.svg`.
- `--plot_only`: panel is replotted from the source data, if it exists. The full analysis is not run.
- `--overwrite`: analysis overwrites panel source data and plot, if they exist. If used with `--plot_only`, however, only the panel plot is overwritten, if it exists.
- `--parallel`: analysis is run using all available CPU cores, to increase code efficiency, and reduce execution time.
- `--datadir`: path to the full dataset (e.g., in NWB format).  
- **Details on additional arguments** can be found in `run_paper_figures.py` or can be printed to the console by running, from the main directory, `python run_paper_figures.py --help`.



<a name="1"><sup>1</sup></a> On a machine with 16+ CPU cores and 150+ GB of RAM, full power analyses for most panels generally took up to 10 minutes each to run, except for Fig. 2B, 7 and 7-1 analyses, which ran in ~1h, ~8h and ~10h each, respectively.
