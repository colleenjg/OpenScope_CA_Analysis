"""
paper_organization.py

This script contains functions and objects for linking analyses to the paper 
structure.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

from collections import namedtuple
from pathlib import Path
import time

from util import gen_util, logger_util, rand_util
from sess_util import sess_gen_util
from paper_fig_util import behav_figs, corr_figs, decoding_figs, misc_figs, \
    seq_figs, stim_figs, roi_figs, usi_figs, plot_figs


PAPER_SEED = 905
DEFAULT_LOW_POWER = 1e3
WARNING_SLEEP = 3

WARNING_TUPLE = namedtuple("WarningsTuple", ["message", "analysis_only"])


logger = logger_util.get_module_logger(name=__name__)


# figure/panel combination that is plottable for both papers
PLOTTABLE = {
    "figure": 2,
    "panel": "C",
}


#############################################
def check_paper(paper="dataset"):
    """
    check_paper()

    Checks and returns the paper type.

    Optional args:
        - paper (str): paper type ('dataset' or 'analysis')
                       default: "dataset"

    Returns:
        - paper (str): paper type
    """

    # create a dummy object with 
    dummy_fig_panel = FigurePanelAnalysis(paper=paper, datadir="", **PLOTTABLE)


    paper = str(paper).lower()
    if paper not in dummy_fig_panel.figure_panel_dict.keys():
        raise ValueError("Only the following paper values are "
            f"accepted: {dummy_fig_panel.figure_panel_dict.keys()}.")

    return paper


#############################################
def get_all_figures(paper="dataset"):
    """
    get_all_figures()

    Returns all figures for the specified paper.

    Optional args:
        - paper (str): paper for which to get all figures 
                       ('dataset' or 'analysis')
                       default: "dataset"

    Returns:
        - all_figures (list): list of figures
    """

    # create a dummy object with 
    dummy_fig_panel = FigurePanelAnalysis(paper=paper, datadir="", **PLOTTABLE)

    paper = check_paper(paper)

    all_figures = dummy_fig_panel.figure_panel_dict[paper].keys()
    
    return all_figures


#############################################
def get_all_panels(paper="dataset", figure=2):
    """
    get_all_panels()

    Returns all panels for a figure.

    Optional args:
        - figure (int or str): figure number (e.g., 2 or '3-1')
                               default: 2
    
    Returns:
        - all_panels (list): list of panels for the figure
    """

    # create a dummy object with 
    dummy_fig_panel = FigurePanelAnalysis(paper=paper, datadir="", **PLOTTABLE)
    
    paper = check_paper(paper)

    figure = str(figure).upper()
    if figure not in dummy_fig_panel.figure_panel_dict[paper].keys():
        raise ValueError("Only the following figure values are "
            f"accepted: {dummy_fig_panel.figure_panel_dict[paper].keys()}.")
    
    all_panels = dummy_fig_panel.figure_panel_dict[paper][figure].keys()
    
    return all_panels


### DEFINE FUNCTIONS RETURNING ERRORS AND WARNING TUPLES
#############################################
def no_plot_fct(reason):
    raise ValueError(f"Cannot plot figure panel as it {reason}.")

def partial_plot_fct_warning(message):
    warning_tuple = WARNING_TUPLE(message, False)
    return warning_tuple

def manual_formatting_warning():
    message = ("Minor manual formatting adjustments may be missing.")
    warning_tuple = WARNING_TUPLE(message, False)
    return warning_tuple

def slow_plot_warning():
    message = ("This figure panel takes longer to plot, as it requires "
        "plotting and rasterizing a large number of items.")
    warning_tuple = WARNING_TUPLE(message, False)
    return warning_tuple

def stats_plot_fct_warning():
    message = ("This figure panel includes statistical analyses. "
        "Statistical markers may not be ideally spaced out, and running full "
        "analyses may take longer.")
    warning_tuple = WARNING_TUPLE(message, False)
    return warning_tuple

def seed_warning(seed):
    message = ("Using a different seed from the one used in "
        f"the paper: {seed}. Results may differ slightly "
        "from published results. To use paper seed, run script with "
        "default seed argument, i.e., '--seed paper'.")
    warning_tuple = WARNING_TUPLE(message, False)
    return warning_tuple

def memory_demand_warning():
    message = ("Analyses for this figure panel have high memory demands. "
        "This may be a problem for machines with small amounts of RAM "
        "(e.g., 16 GB), in which case it is recommended to run the analysis "
        "without the parallel argument ('--parallel').")
    warning_tuple = WARNING_TUPLE(message, True)
    return warning_tuple

def power_warning():
    message = ("Reducing number of permutations/shuffles to reduce "
        "computation time. This weakens statistical power for significance "
        "testing a bit, potentially producing results that differ just "
        "slightly from published results. To reproduce paper results exactly, "
        "run script with the '--full_power' argument.")
    warning_tuple = WARNING_TUPLE(message, True)
    return warning_tuple

def heavy_compute_warning():
    message = ("Analyses for this figure panel are very computationally "
        "intensive! Full power analyses should ONLY be run on a machine with "
        "multiple CPU cores (e.g., 16+) and a substantial amount of RAM "
        "(e.g., 150+ GB), and may still take 1h or more.")
    warning_tuple = WARNING_TUPLE(message, True)
    return warning_tuple

def decoder_warning():
    message = ("The decoder analyses are very computationally intensive! "
        "Full power analyses should ONLY be run on a machine with "
        "multiple CPU cores (e.g., 80+) and a substantial amount of RAM "
        "(e.g., 150+ GB), and may take 10h or more. This is because the "
        "paper code is designed to run in one step, on only one machine.")
    warning_tuple = WARNING_TUPLE(message, True)
    return warning_tuple


### DEFINE FUNCTION COLLECTING FIGURE/PANEL SPECIFIC PARAMETERS
#############################################
def get_specific_params(sess_n="1-3", mouse_n="any", plane="all", line="all", 
                        stimtype="gabors", gabfr=3, gab_ori="any", 
                        visflow_dir="both", pre=0, post=0.6, tails=2, 
                        idx_feature="by_exp", comp="Dori", error="sem", 
                        scale=True, tracked=False, rem_bad=True,
                        roi=True, run=False, pupil=False):
    """
    get_specific_params()

    Returns specific parameters for the analysis.

    Optional args:
        - sess_n (str): 
            which sessions numbers to include
            default: "1-3"
        - mouse_n (str): 
            which mouse number to include
            default: "any"
        - plane (str): 
            which planes to include
            default: "all"
        - line (str): 
            which lines to include
            default: "all"
        - stimtype (str): 
            which stimtypes to include
            default: "gabors"
        - gabfr (str): 
            reference Gabor frame
            default: 3
        - gab_ori (str): 
            Gabor mean orientations to include
            default: "any"
        - visflow_dir (str or list): 
            visual flow direction values ("right", "left", "both")
            default: "both"
        - pre (num): 
            number of seconds before reference to include
            default: 0
        - post (num): 
            number of seconds after reference to include
            default: 0.6
        - tails (str or int): 
            p-value tails (2, "hi" or "lo")
            default: 2
        - idx_feature (str): 
            type of feature to measure indices on
            default: "by_exp"
        - comp (str):
            logistic regression comparison defining classes to decode
            default: "Dori"
        - error (str):
            type of error statistic to calculate for analyses
            default: "sem"
        - scale (bool): 
            whether to use ROI scaling
            default: True
        - tracked (bool):
            if True, only tracked ROIs are included
            default: False
        - rem_bad (bool):
            if True, invalid ROIs are removed, or for pupil/running data, 
            missing data is interpolated
            default: True
        - roi (bool): 
            whether ROI data needs to be loaded
            default: True
        - run (bool): 
            whether run data needs to be loaded
            default: True
        - pupil (bool): 
            whether pupil data needs to be loaded
            default: False

    Returns:
        - specific_params (dict): dictionary with the specified parameters
    """

    visflow_dir, visflow_size, gabfr, gabk, gab_ori = sess_gen_util.get_params(
        stimtype, gabfr=gabfr, gab_ori=gab_ori, visflow_dir=visflow_dir
        )

    specific_params = {
        "sess_n"      : sess_n,
        "mouse_n"     : mouse_n,
        "plane"       : plane,
        "line"        : line,
        "stimtype"    : stimtype,
        "visflow_dir" : visflow_dir,
        "visflow_size": visflow_size,
        "gabfr"       : gabfr,
        "gabk"        : gabk,
        "gab_ori"     : gab_ori,
        "pre"         : pre,
        "post"        : post,
        "tails"       : tails,
        "idx_feature" : idx_feature,
        "comp"        : comp,  
        "error"       : error,
        "scale"       : scale,
        "tracked"     : tracked,
        "rem_bad"     : rem_bad,
        "roi"         : roi,
        "run"         : run,
        "pupil"       : pupil,
    }

    return specific_params


#############################################
class FigurePanelAnalysis():
    def __init__(self, figure, panel, datadir, paper="dataset", 
                 mouse_df_path="mouse_df.csv", output="paper_figures", 
                 full_power=False, parallel=False, seed="paper", plt_bkend=None, 
                 fontdir=None):
        """
        Initializes a FigurePanelAnalysis object.

        Sets attributes from input arguments, and runs self._set_plot_info().

        Required args:
            - figure (str): figure 
            - panel (str): panel
            - datadir (Path): data directory
        
        Optional args:
            - paper (str): 
                paper for which to get all figures ('dataset' or 'analysis')
                default: "dataset"
            - mouse_df_path (Path): 
                mouse dataframe path
                default: "mouse_df.csv"
            - output (Path): 
                output path
                default: "paper_figures"
            - full_power (bool): 
                if True, analyses are run at full statistical power, i.e. with 
                the number of permutations used in the paper
                default: False
            - parallel (bool): 
                if True, some of the analysis is run in parallel across CPU 
                cores 
                default: False
            - seed (int or str): 
                seed to use for random processes
                default: "paper"
            - plt_bkend (str): 
                matplotlib plot backend to use
                default: None
            - fontdir (Path): 
                directory with additional fonts
                default: None
        """
        
        self.paper  = str(paper).lower()
        self.figure = str(figure).upper()
        self.panel  = str(panel).capitalize()

        self.datadir       = Path(datadir)
        self.mouse_df_path = Path(mouse_df_path)
        self.output        = Path(output)
        
        self.parallel  = parallel
        self.plt_bkend = plt_bkend
        self.fontdir   = fontdir

        self.warnings = [manual_formatting_warning()]
        self.n_perms_low = DEFAULT_LOW_POWER
        self.full_power = full_power
        self.seed       = seed

        self._set_plot_info()
        

    def _set_power(self):
        """
        self._set_power()

        Sets and updates attributes related to statistical power for an 
        analysis.

        Sets the following attributes:
            - n_perms (int or list): number of permutations that will be used

        Sets or updates the following attributes:
            - randomness (bool): whether randomness is involved in the analysis

        Updates the following attributes:
            - full_power (bool): 
                whether analysis will be run at full statistic power (same 
                number of permutations as used in the paper)
        """

        if not hasattr(self, "description"):
            raise RuntimeError("Must run self._set_plot_info() first.")

        if not hasattr(self, "randomness"):
            self.randomness = False

        if hasattr(self, "n_perms_full"):
            self.randomness = True
            if self.full_power:
                self.n_perms = self.n_perms_full
            else:
                if self.n_perms_full > self.n_perms_low:
                    self.n_perms = self.n_perms_low
                else:
                    self.n_perms = self.n_perms_full
                    self.full_power = True
            
            self.n_perms = int(self.n_perms)
        else:
            self.full_power = True
            self.n_perms = None

        if not self.full_power:
            self.warnings.append(power_warning())

    def _set_seed(self):
        """
        self._set_seed()

        Updates attributes related to random process seeding.

        Updates the following attributes:
            - paper_seed (bool): whether the paper seed will be used
            - seed (bool): 
                specific seed that will be used None, if self.randomness is 
                False
        """
        
        if not hasattr(self, "n_perms"):
            raise RuntimeError("Must run self._set_power() first.")

        if not self.randomness:
            self.seed = None
            self.paper_seed = True
        else:
            if self.seed == "paper":
                self.seed = PAPER_SEED
                self.paper_seed = True
            else:
                self.seed = int(self.seed)
                self.paper_seed = False
            
            if self.seed == -1:
                self.seed = PAPER_SEED # select any seed but the paper seed
                while self.seed == PAPER_SEED:
                    self.seed = rand_util.seed_all(
                        -1, "cpu", log_seed=False, seed_now=False
                        )
            if self.seed != PAPER_SEED:
                self.warnings.append(seed_warning(self.seed))


    def _set_plot_info(self):
        """
        self._set_plot_info()

        Calls correct function to define figure/panel analysis attributes, as 
        well as self._set_power() and self._set_seed().
        """

        if self.paper not in self.figure_panel_dict.keys():
            raise ValueError("Only the following paper values are accepted: "
            f"{self.figure_panel_dict.keys()}.")

        if self.figure not in self.figure_panel_dict[self.paper].keys():
            raise ValueError("Only the following figure values are "
                f"accepted: {self.figure_panel_dict[self.paper].keys()}.")
        else:
            existing_panels = self.figure_panel_dict[self.paper][self.figure].keys()
            if self.panel not in existing_panels:
                existing_panel_strs = ", ".join(existing_panels)
                raise ValueError(f"Panel {self.panel} is not recognized for "
                    f"figure {self.figure}. Existing panels are "
                    f"{existing_panel_strs}.")

        self.figure_panel_dict[self.paper][self.figure][self.panel]()

        self._set_power()
        self._set_seed()


    def log_warnings(self, plot_only=False):
        """
        Logs figure/panel warning messages, stored in self.warnings, to the 
        console.
        """

        if len(self.warnings):
            messages = [
                warning.message for warning in self.warnings 
                if not (plot_only and warning.analysis_only)
                ]

            warn_str = "\n- " + "\n- ".join(messages)
            
            logger.warning(warn_str, extra={"spacing": "\n"})
            time.sleep(WARNING_SLEEP)


    ### FUNCTIONS DEFINING FIGURE/PANEL ANALYSIS ATTRIBUTES ###
    #############################################
    ### Experiment basics ###
    def structure_schematic(self):
        self.description = ("Schematic illustration of a predictive "
            "hierarchical model.")
        self.specific_params = None
        self.analysis_fct = None
        self.plot_fct = None
        no_plot_fct(reason="is a schematic illustration")


    def imaging_schematic(self):
        self.description = ("Schematic illustration of the experimental "
            "setup.")
        self.specific_params = None
        self.analysis_fct = None
        self.plot_fct = None
        no_plot_fct(reason="is a schematic illustration")


    def imaging_plane_schematic(self):
        self.description = ("Schematic illustration of the four imaging "
            "planes.")
        self.specific_params = None
        self.analysis_fct = None
        self.plot_fct = None
        no_plot_fct(reason="is a schematic illustration")


    def gabor_sequences(self):
        self.description = "Example Gabor sequences."
        self.specific_params = None
        self.analysis_fct = None
        self.plot_fct = None
        no_plot_fct(reason="was drawn manually from images")


    def experimental_timeline(self):
        self.description = "Experimental timeline."
        self.specific_params = None
        self.analysis_fct = None
        self.plot_fct = None
        no_plot_fct(reason="was drawn manually")


    def visual_flow_stimulus(self):
        self.description = "Visual flow stimulus."
        self.specific_params = None
        self.analysis_fct = None
        self.plot_fct = None
        no_plot_fct(reason="was drawn manually from images")


    ### ROI  ###
    def imaging_planes(self):
        self.description = "Example projections of 2p imaging planes."
        self.specific_params = get_specific_params(
            mouse_n=[1, 4, 6, 11], # matches video
            sess_n=1,
            roi=False,
        )
        self.analysis_fct = None
        self.plot_fct = None
        self.analysis_fct = roi_figs.imaging_planes
        self.plot_fct = plot_figs.plot_imaging_planes


    def roi_tracking(self):
        self.description = "Example ROI tracking overlays."
        self.specific_params = get_specific_params(
            mouse_n=[4, 11],
            tracked=True,
        )
        self.analysis_fct = roi_figs.roi_tracking
        self.plot_fct = plot_figs.plot_roi_tracking
        self.warnings.append(memory_demand_warning())


    def roi_overlays_sess123(self):
        self.description = "Example ROI tracking overlays (large)."
        self.specific_params = get_specific_params(
            mouse_n=[3, 4, 6, 11],
            tracked=True,
        )
        self.analysis_fct = roi_figs.roi_overlays_sess123
        self.plot_fct = plot_figs.plot_roi_overlays_sess123
        self.warnings.append(memory_demand_warning())


    def roi_overlays_sess123_enlarged(self):
        self.description = "Example ROI tracking overlay close-ups (large)."
        self.specific_params = get_specific_params(
            mouse_n=[3, 4, 6, 11],
            tracked=True,
        )
        self.analysis_fct = roi_figs.roi_overlays_sess123_enlarged
        self.plot_fct = plot_figs.plot_roi_overlays_sess123_enlarged
        self.warnings.append(memory_demand_warning())

    def dendritic_roi_tracking_example(self):
        self.description = "Dendritic ROI matching examples."
        self.specific_params = get_specific_params(
            plane="dend",
            mouse_n=6,
            tracked=True,
        )
        self.analysis_fct = roi_figs.dendritic_roi_tracking_example
        self.plot_fct = plot_figs.plot_dendritic_roi_tracking_example
        self.warnings.append(memory_demand_warning())


    def somatic_roi_tracking_example(self):
        self.description = "Somatic ROI matching examples."
        self.specific_params = get_specific_params(
            plane="soma",
            mouse_n=4,
            tracked=True,
        )
        self.analysis_fct = roi_figs.somatic_roi_tracking_example
        self.plot_fct = plot_figs.plot_somatic_roi_tracking_example
        self.warnings.append(memory_demand_warning())


    ### Characterization  ###
    def snrs_sess123(self):
        self.description = "Fluorescence SNR for each ROI."
        self.specific_params = get_specific_params(
            scale=False
        )
        self.analysis_fct = misc_figs.snrs_sess123
        self.plot_fct = plot_figs.plot_snrs_sess123
        

    def mean_signal_sess123(self):
        self.description = "Fluorescence signal for each ROI."
        self.specific_params = get_specific_params(
            scale=False
        )
        self.analysis_fct = misc_figs.mean_signal_sess123
        self.plot_fct = plot_figs.plot_mean_signal_sess123
        

    def nrois_sess123(self):
        self.description = "Number of ROIs per session, per mouse."
        self.specific_params = get_specific_params()
        self.analysis_fct = misc_figs.nrois_sess123
        self.plot_fct = plot_figs.plot_nrois_sess123


    def roi_corr_sess123(self):
        self.description = "ROI correlations per session."
        self.specific_params = get_specific_params(
            scale=False
        )
        self.analysis_fct = misc_figs.roi_corr_sess123
        self.plot_fct = plot_figs.plot_roi_corr_sess123


    ### Behaviour ###
    def run_example(self):
        self.description = "Example running frame."
        self.specific_params = None
        self.analysis_fct = None
        self.plot_fct = None
        no_plot_fct(reason="is made up of sample images")


    def pupil_example(self):
        self.description = "Example pupil frames, with markings."
        self.specific_params = None
        self.analysis_fct = None
        self.plot_fct = None
        no_plot_fct(reason="is made up of sample images")


    def pupil_run_responses(self):
        self.description = "Running and pupil responses to Gabor sequences."
        self.specific_params = get_specific_params(
            sess_n=1,
            pre=0.9,
            post=0.6,
            rem_bad=False, # do not interpolate missing data
            scale=False,
            roi=False,
            run=True,
            pupil=True,
        )
        self.analysis_fct = behav_figs.pupil_run_responses
        self.plot_fct = plot_figs.plot_pupil_run_responses
        self.warnings.append(
            partial_plot_fct_warning(
                message="Running and pupil images will be missing."
            )
        )

    def pupil_run_block_diffs(self):
        self.description = ("Trial differences in running and pupil "
            "responses U-G vs D-G Gabor sequences.")
        self.specific_params = get_specific_params(
            sess_n=1,
            rem_bad=False, # do not interpolate missing data
            scale=False,
            roi=False,
            run=True,
            pupil=True,
        )
        self.n_perms_full = 1e4
        self.randomness = True # for plotting, also
        self.analysis_fct = behav_figs.pupil_run_block_diffs
        self.plot_fct = plot_figs.plot_pupil_run_block_diffs


    def pupil_run_full(self):
        self.description = "Full session running and pupil responses."
        self.specific_params = get_specific_params(
            sess_n=1,
            mouse_n=1,
            rem_bad=False, # do not interpolate missing data
            scale=False,
            roi=False,
            run=True,
            pupil=True,
        )
        self.analysis_fct = behav_figs.pupil_run_full
        self.plot_fct = plot_figs.plot_pupil_run_full
        self.warnings.append(
            partial_plot_fct_warning(
                message="Running and pupil images will be missing."
            )
        )

    def pupil_run_histograms(self):
        self.description = "Histograms of running and pupil values."
        self.specific_params = get_specific_params(
            rem_bad=False, # do not interpolate missing data
            scale=False,
            roi=False,
            run=True,
            pupil=True,
        )
        self.analysis_fct = behav_figs.pupil_run_histograms
        self.plot_fct = plot_figs.plot_pupil_run_histograms


    ### General responses ###
    def stimulus_onset_sess123(self):
        self.description = "ROI response to stimulus onset."
        self.specific_params = get_specific_params(
            stimtype="both",
            pre=2,
            post=2,
        )
        self.analysis_fct = seq_figs.stimulus_onset_sess123
        self.plot_fct = plot_figs.plot_stimulus_onset_sess123


    def stimulus_offset_sess123(self):
        self.description = "ROI response to stimulus offset."
        self.specific_params = get_specific_params(
            stimtype="both",
            pre=2,
            post=2,
        )
        self.analysis_fct = seq_figs.stimulus_offset_sess123
        self.plot_fct = plot_figs.plot_stimulus_offset_sess123


    def gabor_ex_roi_exp_responses_sess1(self):
        self.description = (
            "Example ROI responses to each consistent Gabor sequence." # dataset paper
            )
        self.specific_params = get_specific_params(
            sess_n=1,
            pre=0.9,
            post=0.6,
            )
        self.randomness = True # for example selection
        self.analysis_fct = seq_figs.gabor_ex_roi_exp_responses_sess1
        self.plot_fct = plot_figs.plot_gabor_ex_roi_exp_responses_sess1
        self.warnings.append(slow_plot_warning())


    def gabor_ex_roi_unexp_responses_sess1(self):
        self.description = (
            "Example ROI responses to each inconsistent Gabor sequence." # dataset paper
            )
        self.specific_params = get_specific_params(
            sess_n=1,
            pre=0.9,
            post=0.6,
            )
        self.randomness = True # for example selection
        self.analysis_fct = seq_figs.gabor_ex_roi_unexp_responses_sess1
        self.plot_fct = plot_figs.plot_gabor_ex_roi_unexp_responses_sess1
        self.warnings.append(slow_plot_warning())


    def visflow_ex_roi_nasal_responses_sess1(self):
        self.description = (
            "Example ROI responses to each onset of inconsistent flow during " # dataset paper
            "nasal (leftward) visual flow."
            )
        self.specific_params = get_specific_params(
            sess_n=1,
            stimtype="visflow",
            pre=2,
            post=2,
            visflow_dir="nasal",
            )
        self.randomness = True # for example selection
        self.analysis_fct = seq_figs.visflow_ex_roi_nasal_responses_sess1
        self.plot_fct = plot_figs.plot_visflow_ex_roi_nasal_responses_sess1
        self.warnings.append(slow_plot_warning())


    def visflow_ex_roi_temp_responses_sess1(self):
        self.description = (
            "Example ROI responses to each onset of inconsistent flow during " # dataset paper
            "temporal (rightward) visual flow."
            )
        self.specific_params = get_specific_params(
            sess_n=1,
            stimtype="visflow",
            pre=2,
            post=2,
            visflow_dir="temp",
            )
        self.randomness = True # for example selection
        self.analysis_fct = seq_figs.visflow_ex_roi_temp_responses_sess1
        self.plot_fct = plot_figs.plot_visflow_ex_roi_temp_responses_sess1
        self.warnings.append(slow_plot_warning())


    ### Sequences ###
    def gabor_sequences_sess123(self):
        self.description = "ROI responses to Gabor sequences."
        self.specific_params = get_specific_params(
            pre=0.9,
        )
        self.analysis_fct = seq_figs.gabor_sequences_sess123
        self.plot_fct = plot_figs.plot_gabor_sequences_sess123


    def gabor_sequence_diffs_sess123(self):
        self.description = ("Differences in ROI responses to unexpected "
            "and expected Gabor sequences.")
        self.specific_params = get_specific_params(
            pre=0.9,
        )
        self.n_perms_full = 1e5
        self.analysis_fct = seq_figs.gabor_sequence_diffs_sess123
        self.plot_fct = plot_figs.plot_gabor_sequence_diffs_sess123
        self.warnings.append(stats_plot_fct_warning())
        self.warnings.append(heavy_compute_warning())
        

    def gabor_sequence_early_late_sess123(self):
        self.description = ("ROI responses to Gabor sequences, "
            "early and late in sessions.")
        self.specific_params = get_specific_params(
            pre=0.9,
        )
        self.analysis_fct = seq_figs.gabor_sequence_early_late_sess123
        self.plot_fct = plot_figs.plot_gabor_sequences_early_late_sess123


    def gabor_sequence_diffs_early_late_sess123(self):
        self.description = ("Differences in ROI responses to unexpected "
            "and expected Gabor sequences, early and late in sessions.")
        self.specific_params = get_specific_params(
            pre=0.9,
        )
        self.n_perms_full = 1e5
        self.analysis_fct = seq_figs.gabor_sequence_diffs_early_late_sess123
        self.plot_fct = plot_figs.plot_gabor_sequence_diffs_early_late_sess123
        self.warnings.append(stats_plot_fct_warning())
        

    def gabor_rel_resp_sess123(self):
        self.description = ("ROI responses to regular and unexpected "
            "Gabor sequences, relative to session 1.")
        self.specific_params = get_specific_params(
            pre=0,
            post=0.3,
            gabfr=[[0, 1, 2], [3, 4]],
            scale=False,
        )
        self.n_perms_full = 1e5
        self.analysis_fct = seq_figs.gabor_rel_resp_sess123
        self.plot_fct = plot_figs.plot_gabor_rel_resp_sess123
        self.warnings.append(stats_plot_fct_warning())


    def visual_flow_sequences_sess123(self):
        self.description = "ROI responses to visual flow sequences."
        self.specific_params = get_specific_params(
            stimtype="visflow",
            pre=2,
            post=2,
        )
        self.analysis_fct = seq_figs.visual_flow_sequences_sess123
        self.plot_fct = plot_figs.plot_visual_flow_sequences_sess123
        

    def visual_flow_diffs_sess123(self):
        self.description = ("Differences in ROI responses to unexpected "
            "and expected visual flow sequences.")
        self.specific_params = get_specific_params(
            stimtype="visflow",
            pre=2,
            post=2,
        )
        self.n_perms_full = 1e5
        self.analysis_fct = seq_figs.visual_flow_diffs_sess123
        self.plot_fct = plot_figs.plot_visual_flow_diffs_sess123
        self.warnings.append(stats_plot_fct_warning())


    def visual_flow_rel_resp_sess123(self):
        self.description = ("ROI responses to expected and unexpected "
            "visual flow sequences, relative to session 1.")
        self.specific_params = get_specific_params(
            stimtype="visflow",
            pre=0,
            post=1,
            scale=False,
        )
        self.n_perms_full = 1e5
        self.analysis_fct = seq_figs.visual_flow_rel_resp_sess123
        self.plot_fct = plot_figs.plot_visual_flow_rel_resp_sess123
        self.warnings.append(stats_plot_fct_warning())


    ### USIs ###
    def gabor_example_roi_usis(self):
        self.description = "Example ROI responses to Gabor sequences."
        self.specific_params = get_specific_params(
            sess_n=1,
            pre=[0, 0.9],
        )
        self.n_perms_full = 5e3
        self.analysis_fct = usi_figs.gabor_example_roi_usis
        self.plot_fct = plot_figs.plot_gabor_example_roi_usis
        
        
    def gabor_example_roi_usi_sig(self):
        self.description = ("Example Gabor USI null distribution for a "
            "single ROI.")
        self.specific_params = get_specific_params(
            sess_n=1,
            mouse_n=1,
            line="L23",
            plane="soma",
        )
        self.n_perms_full = 5e3
        self.analysis_fct = usi_figs.gabor_example_roi_usi_sig
        self.plot_fct = plot_figs.plot_gabor_example_roi_usi_sig


    def gabor_roi_usi_distr(self):
        self.description = "Distributions of Gabor USI percentiles."
        self.specific_params = get_specific_params(
            sess_n=1,
        )
        self.n_perms_full = 5e3
        self.analysis_fct = usi_figs.gabor_roi_usi_distr
        self.plot_fct = plot_figs.plot_gabor_roi_usi_distr


    def gabor_roi_usi_sig(self):
        self.description = "Percentages of significant Gabor USIs."
        self.specific_params = get_specific_params(
            sess_n=1,
            error="std",
        )
        self.n_perms_full = 1e4
        self.analysis_fct = usi_figs.gabor_roi_usi_sig
        self.plot_fct = plot_figs.plot_gabor_roi_usi_sig
        self.warnings.append(stats_plot_fct_warning())


    def gabor_roi_usi_sig_common_oris(self):
        self.description = ("Percentages of significant Gabor USIs for "
            "sequences with orientations common to D/U.")
        self.specific_params = get_specific_params(
            sess_n=1,
            error="std",
        )
        self.n_perms_full = 1e4
        self.analysis_fct = usi_figs.gabor_roi_usi_sig_common_oris
        self.plot_fct = plot_figs.plot_gabor_roi_usi_sig_common_oris
        self.warnings.append(stats_plot_fct_warning())

    def gabor_roi_usi_sig_by_mouse(self):
        self.description = ("Percentages of significant Gabor USIs for "
            "each mouse.")
        self.specific_params = get_specific_params(
            sess_n=1,
            error="std",
        )
        self.n_perms_full = 1e4
        self.analysis_fct = usi_figs.gabor_roi_usi_sig_by_mouse
        self.plot_fct = plot_figs.plot_gabor_roi_usi_sig_by_mouse
        self.warnings.append(stats_plot_fct_warning())


    def gabor_rel_resp_tracked_rois_sess123(self):
        self.description = ("Tracked ROI responses to regular and unexpected "
            "Gabor sequences, relative to session 1.")
        self.specific_params = get_specific_params(
            pre=0,
            post=0.3,
            gabfr=[[0, 1, 2], [3, 4]],
            scale=False,
            tracked=True,
        )
        self.n_perms_full = 1e5
        self.analysis_fct = seq_figs.gabor_rel_resp_tracked_rois_sess123
        self.plot_fct = plot_figs.plot_gabor_rel_resp_tracked_rois_sess123
        self.warnings.append(stats_plot_fct_warning())


    def gabor_tracked_roi_abs_usi_means_sess123_by_mouse(self):
        self.description = ("Absolute means of tracked ROI Gabor USIs "
            "across sessions for each mouse.")
        self.specific_params = get_specific_params(
            tracked=True
        )
        self.analysis_fct = usi_figs.gabor_tracked_roi_abs_usi_means_sess123_by_mouse
        self.plot_fct = plot_figs.plot_gabor_tracked_roi_abs_usi_means_sess123_by_mouse
        self.warnings.append(stats_plot_fct_warning())


    ### Tracked USIs ###
    def gabor_tracked_roi_usis_sess123(self):
        self.description = "Tracked ROI Gabor USIs across sessions."
        self.specific_params = get_specific_params(
            tracked=True
        )
        self.analysis_fct = usi_figs.gabor_tracked_roi_usis_sess123
        self.plot_fct = plot_figs.plot_gabor_tracked_roi_usis_sess123
        

    def gabor_tracked_roi_abs_usi_means_sess123(self):
        self.description = ("Absolute means of tracked ROI Gabor USIs "
            "across sessions.")
        self.specific_params = get_specific_params(
            tracked=True
        )
        self.n_perms_full = 1e5
        self.analysis_fct = usi_figs.gabor_tracked_roi_abs_usi_means_sess123
        self.plot_fct = plot_figs.plot_gabor_tracked_roi_abs_usi_means_sess123
        self.warnings.append(stats_plot_fct_warning())


    def gabor_tracked_roi_usi_variances_sess123(self):
        self.description = ("Variances of tracked ROI Gabor USIs "
            "across sessions.")
        self.specific_params = get_specific_params(
            tracked=True,
            error="std",
        )
        self.n_perms_full = 1e5
        self.analysis_fct = usi_figs.gabor_tracked_roi_usi_variances_sess123
        self.plot_fct = plot_figs.plot_gabor_tracked_roi_usi_variances_sess123
        self.warnings.append(stats_plot_fct_warning())

    def gabor_corrs_sess123_comps(self):
        self.description = "Gabor USI correlations between sessions."
        self.specific_params = get_specific_params(
            error="std",
            tracked=True,
            tails="lo",
        )
        self.n_perms_full = 1e5
        self.analysis_fct = corr_figs.gabor_corrs_sess123_comps
        self.plot_fct = plot_figs.plot_gabor_corrs_sess123_comps
        self.warnings.append(stats_plot_fct_warning())

    def gabor_corr_scatterplots_sess12(self):
        self.description = "Gabor USI session 1 vs 2 correlation scatterplots."
        self.specific_params = get_specific_params(
            sess_n="1-2",
            error="std",
            tracked=True,
            tails="lo",
        )
        self.n_perms_full = 1e5
        self.analysis_fct = corr_figs.gabor_corr_scatterplots_sess12
        self.plot_fct = plot_figs.plot_gabor_corr_scatterplots_sess12


    def gabor_corr_scatterplots_sess23(self):
        self.description = "Gabor USI session 2 vs 3 correlation scatterplots."
        self.specific_params = get_specific_params(
            sess_n="2-3",
            error="std",
            tracked=True,
            tails="lo",
        )
        self.n_perms_full = 1e5
        self.analysis_fct = corr_figs.gabor_corr_scatterplots_sess23
        self.plot_fct = plot_figs.plot_gabor_corr_scatterplots_sess23


    ### Visflow USIs ###
    def visual_flow_tracked_roi_usis_sess123(self):
        self.description = "Tracked ROI visual flow USIs across sessions."
        self.specific_params = get_specific_params(
            stimtype="visflow",
            pre=2,
            post=2,
            idx_feature="unexp_lock",
            tracked=True,
        )
        self.analysis_fct = usi_figs.visual_flow_tracked_roi_usis_sess123
        self.plot_fct = plot_figs.plot_visual_flow_tracked_roi_usis_sess123
        

    def visual_flow_tracked_roi_abs_usi_means_sess123(self):
        self.description = ("Absolute means of tracked ROI visual flow USIs "
            "across sessions.")
        self.specific_params = get_specific_params(
            stimtype="visflow",
            pre=2,
            post=2,
            idx_feature="unexp_lock",
            tracked=True,
        )
        self.n_perms_full = 1e5
        self.analysis_fct = usi_figs.visual_flow_tracked_roi_abs_usi_means_sess123
        self.plot_fct = plot_figs.plot_visual_flow_tracked_roi_abs_usi_means_sess123
        self.warnings.append(stats_plot_fct_warning())
        

    def visual_flow_corrs_sess123_comps(self):
        self.description = "Visual flow USI correlations between sessions."
        self.specific_params = get_specific_params(
            stimtype="visflow",
            pre=2,
            post=2,
            idx_feature="unexp_lock",
            error="std",
            tails="lo",
            tracked=True,
        )
        self.n_perms_full = 1e5
        self.analysis_fct = corr_figs.visual_flow_corrs_sess123_comps
        self.plot_fct = plot_figs.plot_visual_flow_corrs_sess123_comps


    def visual_flow_corr_scatterplots_sess12(self):
        self.description = "Visual flow USI session 1 vs 2 correlation scatterplots."
        self.specific_params = get_specific_params(
            sess_n="1-2",
            stimtype="visflow",
            pre=2,
            post=2,
            idx_feature="unexp_lock",
            error="std",
            tails="lo",
            tracked=True,
        )
        self.n_perms_full = 1e5
        self.analysis_fct = corr_figs.visual_flow_corr_scatterplots_sess12
        self.plot_fct = plot_figs.plot_visual_flow_corr_scatterplots_sess12
        

    def visual_flow_corr_scatterplots_sess23(self):
        self.description = "Visual flow USI session 2 vs 3 correlation scatterplots."
        self.specific_params = get_specific_params(
            sess_n="2-3",
            stimtype="visflow",
            pre=2,
            post=2,
            idx_feature="unexp_lock",
            error="std",
            tails="lo",
            tracked=True,
        )
        self.n_perms_full = 1e5
        self.analysis_fct = corr_figs.visual_flow_corr_scatterplots_sess23
        self.plot_fct = plot_figs.plot_visual_flow_corr_scatterplots_sess23


    ### Stimulus comparisons ###
    def unexp_resp_stimulus_comp_sess1v3(self):
        self.description = ("Change in ROI responses to unexpected sequences "
            "for the Gabor vs visual flow stimulus.")
        self.specific_params = get_specific_params(
            stimtype=["gabors", "visflow"],
            pre=[0, 0],
            post=[0.3, 1],
            gabfr=[[0, 1, 2], [3, 4]],
            scale=False,
            error="std",
        ) 
        self.n_perms_full = 1e5
        self.analysis_fct = stim_figs.unexp_resp_stimulus_comp_sess1v3
        self.plot_fct = plot_figs.plot_unexp_resp_stimulus_comp_sess1v3
        self.warnings.append(stats_plot_fct_warning())


    def tracked_roi_usis_stimulus_comp_sess1v3(self):
        self.description = ("Change in tracked ROI USIs for the Gabor vs "
            "visual flow stimulus.")
        self.specific_params = get_specific_params(
            stimtype=["gabors", "visflow"],
            pre=[0, 2],
            post=[0.6, 2],
            idx_feature=["by_exp", "unexp_lock"],
            error="std",
            tracked=True,
        ) 
        self.n_perms_full = 1e5
        self.analysis_fct = stim_figs.tracked_roi_usis_stimulus_comp_sess1v3
        self.plot_fct = plot_figs.plot_tracked_roi_usis_stimulus_comp_sess1v3


    ### Decoding ### 
    def gabor_Aori_decoding_sess123(self):
        self.description = ("Mean Gabor A orientation decoder performances "
            "across sessions.")
        self.specific_params = get_specific_params(
            comp="Aori",
            gabfr=0,
            post=0.45,
            tails="hi",
        )
        self.n_perms_full = 1e5
        self.n_perms_low = 2500
        self.analysis_fct = decoding_figs.gabor_Aori_decoding_sess123
        self.plot_fct = plot_figs.plot_gabor_Aori_decoding_sess123
        self.warnings.append(stats_plot_fct_warning())
        self.warnings.append(decoder_warning())


    def gabor_Bori_decoding_sess123(self):
        self.description = ("Mean Gabor B orientation decoder performances "
            "across sessions.")
        self.specific_params = get_specific_params(
            comp="Bori",
            gabfr=1,
            post=0.45,
            tails="hi",
        )
        self.n_perms_full = 1e5
        self.n_perms_low = 2500
        self.analysis_fct = decoding_figs.gabor_Bori_decoding_sess123
        self.plot_fct = plot_figs.plot_gabor_Bori_decoding_sess123
        self.warnings.append(stats_plot_fct_warning())
        self.warnings.append(decoder_warning())


    def gabor_Cori_decoding_sess123(self):
        self.description = ("Mean Gabor C orientation decoder performances "
            "across sessions.")
        self.specific_params = get_specific_params(
            comp="Cori",
            gabfr=2,
            post=0.45,
            tails="hi",
        )
        self.n_perms_full = 1e5
        self.n_perms_low = 2500
        self.analysis_fct = decoding_figs.gabor_Cori_decoding_sess123
        self.plot_fct = plot_figs.plot_gabor_Cori_decoding_sess123
        self.warnings.append(stats_plot_fct_warning())
        self.warnings.append(decoder_warning())

    def gabor_Dori_decoding_sess123(self):
        self.description = ("Mean Gabor D orientation decoder performances "
            "across sessions.")
        self.specific_params = get_specific_params(
            comp="Dori",
            gabfr=3,
            post=0.45,
            tails="hi",
        )
        self.n_perms_full = 1e5
        self.n_perms_low = 2500
        self.analysis_fct = decoding_figs.gabor_Dori_decoding_sess123
        self.plot_fct = plot_figs.plot_gabor_Dori_decoding_sess123
        self.warnings.append(stats_plot_fct_warning()) 
        self.warnings.append(decoder_warning())


    def gabor_Uori_decoding_sess123(self):
        self.description = ("Mean Gabor U orientation decoder performances "
            "across sessions.")
        self.specific_params = get_specific_params(
            comp="Uori",
            gabfr=3,
            post=0.45,
            tails="hi",
        )
        self.n_perms_full = 1e5
        self.n_perms_low = 2500
        self.analysis_fct = decoding_figs.gabor_Uori_decoding_sess123
        self.plot_fct = plot_figs.plot_gabor_Uori_decoding_sess123
        self.warnings.append(stats_plot_fct_warning())
        self.warnings.append(decoder_warning())


    def gabor_timecourse_decoding_sess1(self):
        self.description = ("Mean Gabor timecourse orientation decoder "
            "performances for session 1.")
        self.specific_params = get_specific_params(
            comp=["Dori", "Uori"],
            gabfr=3,
            pre=0.9,
            post=0.6,
            sess_n=1,
            tails="hi",
        )
        self.n_perms_full = 1e4
        self.n_perms_low = 350
        self.analysis_fct = decoding_figs.gabor_timecourse_decoding_sess1
        self.plot_fct = plot_figs.plot_gabor_timecourse_decoding_sess1
        self.warnings.append(stats_plot_fct_warning())
        self.warnings.append(decoder_warning())


    def gabor_timecourse_decoding_sess2(self):
        self.description = ("Mean Gabor timecourse orientation decoder "
            "performances for session 2.")
        self.specific_params = get_specific_params(
            comp=["Dori", "Uori"],
            gabfr=3,
            pre=0.9,
            post=0.6,
            sess_n=2,
            tails="hi",
        )
        self.n_perms_full = 1e4
        self.n_perms_low = 350
        self.analysis_fct = decoding_figs.gabor_timecourse_decoding_sess2
        self.plot_fct = plot_figs.plot_gabor_timecourse_decoding_sess2
        self.warnings.append(stats_plot_fct_warning())
        self.warnings.append(decoder_warning())


    def gabor_timecourse_decoding_sess3(self):
        self.description = ("Mean Gabor timecourse orientation decoder "
            "performances for session 3.")
        self.specific_params = get_specific_params(
            comp=["Dori", "Uori"],
            gabfr=3,
            pre=0.9,
            post=0.6,
            sess_n=3,
            tails="hi",
        )
        self.n_perms_full = 1e4
        self.n_perms_low = 350
        self.analysis_fct = decoding_figs.gabor_timecourse_decoding_sess3
        self.plot_fct = plot_figs.plot_gabor_timecourse_decoding_sess3
        self.warnings.append(stats_plot_fct_warning())
        self.warnings.append(decoder_warning())

    ### Model ###
    def model_illustration(self):
        self.description = ("Schematic illustration of a conceptual model "
            "based on the data.")
        self.specific_params = None
        self.analysis_fct = None
        self.plot_fct = None
        no_plot_fct(reason="is a schematic illustration")


    @property
    def figure_panel_dict(self):
        """
        Sets and returns self._figure_panel_dict attribute, which maps figure 
        panels to the correct function to define analysis attributes. 
        """
        if not hasattr(self, "_figure_panel_dict"):
            self._figure_panel_dict = {
                "dataset": {
                    "1": {
                        "A": self.imaging_schematic,
                        "B": self.imaging_planes,
                        "C": self.imaging_plane_schematic,
                        },
                    "2": {
                        "A": self.experimental_timeline,
                        "B": self.roi_tracking,
                        "C": self.roi_overlays_sess123,
                        "D": self.roi_overlays_sess123_enlarged
                        },
                    "3": {
                        "A": self.gabor_sequences,
                        "B": self.visual_flow_stimulus,
                        },
                    "4": {
                        "A": self.run_example,
                        "B": self.pupil_example,
                        },
                    "5": {
                        "A": self.gabor_ex_roi_exp_responses_sess1,
                        "B": self.gabor_ex_roi_unexp_responses_sess1,
                        "C": self.visflow_ex_roi_nasal_responses_sess1,
                        "D": self.visflow_ex_roi_temp_responses_sess1,
                        },
                    "6": {
                        "A": self.pupil_run_full,
                        "B": self.pupil_run_histograms,
                        },
                    "7": {
                        "A": self.snrs_sess123,
                        "B": self.mean_signal_sess123,
                        "C": self.roi_corr_sess123,
                        },
                    "8":  {
                        "A": self.dendritic_roi_tracking_example,
                        "B": self.somatic_roi_tracking_example,
                        },
                    "9": {
                        "A": self.stimulus_onset_sess123,
                        "B": self.stimulus_offset_sess123,
                    }
                },

                "analysis": {
                    "1": {
                        "A": self.imaging_schematic,
                        "B": self.imaging_planes,
                        "C": self.imaging_plane_schematic,
                        "D": self.roi_tracking,
                        "E": self.gabor_sequences,
                        "F": self.experimental_timeline,
                        },
                    "2": {
                        "A": self.gabor_sequences_sess123,
                        "B": self.gabor_sequence_diffs_sess123,
                        "C": self.gabor_rel_resp_sess123,
                        },
                    "2-1": {
                        "A": self.gabor_sequence_early_late_sess123,
                        "B": self.gabor_sequence_diffs_early_late_sess123,
                        },
                    "3": {
                        "A": self.gabor_example_roi_usis,
                        "B": self.gabor_example_roi_usi_sig,
                        "C": self.gabor_roi_usi_distr,
                        "D": self.gabor_roi_usi_sig,
                        "E": self.gabor_roi_usi_sig_common_oris,
                        "F": self.gabor_tracked_roi_usis_sess123,
                        "G": self.gabor_tracked_roi_abs_usi_means_sess123,
                        "H": self.gabor_tracked_roi_usi_variances_sess123,
                        },
                    "3-1": {
                        "A": self.gabor_roi_usi_sig_by_mouse,
                        "B": self.gabor_rel_resp_tracked_rois_sess123,
                        "C": self.gabor_tracked_roi_abs_usi_means_sess123_by_mouse,
                        },
                    "4": {
                        "A": self.visual_flow_stimulus,
                        "B": self.visual_flow_sequences_sess123,
                        "C": self.visual_flow_diffs_sess123,
                        },
                    "4-1":  {
                        "A": self.visual_flow_rel_resp_sess123,
                        "B": self.unexp_resp_stimulus_comp_sess1v3,
                        "C": self.visual_flow_tracked_roi_usis_sess123,
                        "D": self.visual_flow_tracked_roi_abs_usi_means_sess123,
                        "E": self.tracked_roi_usis_stimulus_comp_sess1v3,
                        },
                    "5": {
                        "A": self.pupil_run_responses,
                        "B": self.pupil_run_block_diffs,
                        },
                    "6":  {
                        "A": self.gabor_corr_scatterplots_sess12,
                        "B": self.gabor_corr_scatterplots_sess23,
                        "C": self.gabor_corrs_sess123_comps,
                        "D": self.visual_flow_corr_scatterplots_sess12,
                        "E": self.visual_flow_corr_scatterplots_sess23,
                        "F": self.visual_flow_corrs_sess123_comps,
                        },
                    "7": {
                        "A": self.gabor_Aori_decoding_sess123,
                        "B": self.gabor_Bori_decoding_sess123,
                        "C": self.gabor_Cori_decoding_sess123,
                        "D": self.gabor_Dori_decoding_sess123,
                        "E": self.gabor_Uori_decoding_sess123,
                        },
                    "7-1": {
                        "A": self.gabor_timecourse_decoding_sess1,
                        "B": self.gabor_timecourse_decoding_sess2,
                        "C": self.gabor_timecourse_decoding_sess3,
                        },
                    }
                }
        return self._figure_panel_dict

