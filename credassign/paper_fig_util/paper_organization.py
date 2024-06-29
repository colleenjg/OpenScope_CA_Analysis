"""
paper_organization.py

This script contains functions and objects for linking analyses to the paper 
structure.

Authors: Colleen Gillon

Date: February 2023

Note: this code was aggregated from https://github.com/colleenjg/OpenScope_CA_Analysis.
"""

from collections import namedtuple
from pathlib import Path
import time

from credassign.paper_fig_util import analyse_figs, plot_figs
from credassign.util import gen_util, logger_util, sess_util

DEFAULT_MOUSE_DF_PATH = Path(Path(__file__).parent.parent, "mouse_df.csv")

PAPER_SEED = 905
WARNING_SLEEP = 3

WARNING_TUPLE = namedtuple("WarningsTuple", ["message", "analysis_only"])


logger = logger_util.get_module_logger(name=__name__)


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

    if paper != "dataset":
        raise NotImplementedError("Only 'dataset' paper is implemented.")

    # create a dummy object with figure/panel combination that is plottable.
    dummy_fig_panel = FigurePanelAnalysis(
        figure=2, panel="C", paper=paper, datadir=""
        )

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

    # create a dummy object with figure/panel combination that is plottable.
    dummy_fig_panel = FigurePanelAnalysis(
        figure=2, panel="C", paper=paper, datadir=""
        )

    paper = check_paper(paper)

    all_figures = dummy_fig_panel.figure_panel_dict[paper].keys()
    
    return all_figures


#############################################
def get_all_panels(paper="dataset", figure=2):
    """
    get_all_panels()

    Returns all panels for a figure.

    Optional args:
        - figure (int or str): figure number (e.g., 2 or 'S2')
                               default: 2
    
    Returns:
        - all_panels (list): list of panels for the figure
    """

    # create a dummy object with figure/panel combination that is plottable.
    dummy_fig_panel = FigurePanelAnalysis(
        figure=2, panel="C", datadir="", paper=paper
        )
    
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


### DEFINE FUNCTION COLLECTING FIGURE/PANEL SPECIFIC PARAMETERS
#############################################
def get_specific_params(sess_n="1-3", mouse_n="any", plane="all", line="all", 
                        stimtype="gabors", gabfr=3, gab_ori="any", 
                        visflow_dir="both", pre=0, post=0.6, rem_bad=True,
                        error="sem", scale=True, tracked=False, roi=True, 
                        run=False, pupil=False):
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
        - rem_bad (bool):
            if True, missing values are interpolated back in for analysis
            default: True
        - error (str):
            type of error statistic to calculate for analyses
            default: "sem"
        - scale (bool): 
            whether to use ROI scaling
            default: True
        - tracked (bool):
            if True, only tracked ROIs are included
            default: False
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

    visflow_dir, visflow_size, gabfr, gabk, gab_ori = sess_util.get_params(
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
        "rem_bad"     : rem_bad,
        "error"       : error,
        "scale"       : scale,
        "tracked"     : tracked,
        "roi"         : roi,
        "run"         : run,
        "pupil"       : pupil,
    }

    return specific_params


#############################################
class FigurePanelAnalysis():
    def __init__(self, figure, panel, datadir, paper="dataset", 
                 mouse_df_path=DEFAULT_MOUSE_DF_PATH, output="paper_figures", 
                 parallel=False, seed="paper", plt_bkend=None):
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
                default: DEFAULT_MOUSE_DF_PATH
            - output (Path): 
                output path
                default: "paper_figures"
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
        """
        
        self.paper  = str(paper).lower()
        self.figure = str(figure).upper()
        self.panel  = str(panel).capitalize()

        self.datadir       = Path(datadir)
        self.mouse_df_path = Path(mouse_df_path)
        self.output        = Path(output)
        
        self.parallel  = parallel
        self.plt_bkend = plt_bkend

        self.warnings = [manual_formatting_warning()]
        self.seed       = seed

        self._set_plot_info()
        

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

        if not hasattr(self, "description"):
            raise RuntimeError("Must run self._set_plot_info() first.")

        if not hasattr(self, "randomness"):
            self.randomness = False

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
                    self.seed = gen_util.seed_all(-1)
            if self.seed != PAPER_SEED:
                self.warnings.append(seed_warning(self.seed))


    def _set_plot_info(self):
        """
        self._set_plot_info()

        Calls correct function to define figure/panel analysis attributes, as 
        well as self._set_seed().
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
    ### Figure 1 ###
    def imaging_schematic(self):
        self.description = ("Schematic illustration of the experimental "
            "setup.")
        self.specific_params = None
        self.analysis_fct = None
        self.plot_fct = None
        no_plot_fct(reason="is a schematic illustration")


    def imaging_planes(self):
        self.description = "Example projections of 2p imaging planes."
        self.specific_params = get_specific_params(
            mouse_n=[1, 4, 6, 11],
            sess_n=1,
            roi=False,
        )
        self.analysis_fct = None
        self.plot_fct = None
        self.analysis_fct = analyse_figs.imaging_planes
        self.plot_fct = plot_figs.plot_imaging_planes


    def imaging_plane_schematic(self):
        self.description = ("Schematic illustration of the four imaging "
            "planes.")
        self.specific_params = None
        self.analysis_fct = None
        self.plot_fct = None
        no_plot_fct(reason="is a schematic illustration")


    ### Figure 2 ###
    def experimental_timeline(self):
        self.description = "Experimental timeline."
        self.specific_params = None
        self.analysis_fct = None
        self.plot_fct = None
        no_plot_fct(reason="was drawn manually")


    def roi_tracking(self):
        self.description = "Example ROI tracking overlays."
        self.specific_params = get_specific_params(
            mouse_n=[4, 11],
            tracked=True,
        )
        self.analysis_fct = analyse_figs.roi_tracking
        self.plot_fct = plot_figs.plot_roi_tracking
        self.warnings.append(memory_demand_warning())


    def roi_overlays_sess123(self):
        self.description = "Example ROI tracking overlays (large)."
        self.specific_params = get_specific_params(
            mouse_n=[3, 4, 6, 11],
            tracked=True,
        )
        self.analysis_fct = analyse_figs.roi_overlays_sess123
        self.plot_fct = plot_figs.plot_roi_overlays_sess123
        self.warnings.append(memory_demand_warning())


    def roi_overlays_sess123_enlarged(self):
        self.description = "Example ROI tracking overlay close-ups (large)."
        self.specific_params = get_specific_params(
            mouse_n=[3, 4, 6, 11],
            tracked=True,
        )
        self.analysis_fct = analyse_figs.roi_overlays_sess123_enlarged
        self.plot_fct = plot_figs.plot_roi_overlays_sess123_enlarged
        self.warnings.append(memory_demand_warning())


    ### Figure 3 ###
    def gabor_sequences(self):
        self.description = "Example Gabor sequences."
        self.specific_params = None
        self.analysis_fct = None
        self.plot_fct = None
        no_plot_fct(reason="was drawn manually from images")


    def visual_flow_stimulus(self):
        self.description = "Visual flow stimulus."
        self.specific_params = None
        self.analysis_fct = None
        self.plot_fct = None
        no_plot_fct(reason="was drawn manually from images")


    ### Figure 4 ###
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


    ### Figure 5 ###
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
        self.analysis_fct = analyse_figs.gabor_ex_roi_exp_responses_sess1
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
        self.analysis_fct = analyse_figs.gabor_ex_roi_unexp_responses_sess1
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
        self.analysis_fct = analyse_figs.visflow_ex_roi_nasal_responses_sess1
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
        self.analysis_fct = analyse_figs.visflow_ex_roi_temp_responses_sess1
        self.plot_fct = plot_figs.plot_visflow_ex_roi_temp_responses_sess1
        self.warnings.append(slow_plot_warning())


    ### Figure 6 ###
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
        self.analysis_fct = analyse_figs.pupil_run_full
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
        self.analysis_fct = analyse_figs.pupil_run_histograms
        self.plot_fct = plot_figs.plot_pupil_run_histograms

    
    ### Figure 7 ###
    def snrs_sess123(self):
        self.description = "Fluorescence SNR for each ROI."
        self.specific_params = get_specific_params(
            scale=False
        )
        self.analysis_fct = analyse_figs.snrs_sess123
        self.plot_fct = plot_figs.plot_snrs_sess123
        

    def mean_signal_sess123(self):
        self.description = "Fluorescence signal for each ROI."
        self.specific_params = get_specific_params(
            scale=False
        )
        self.analysis_fct = analyse_figs.mean_signal_sess123
        self.plot_fct = plot_figs.plot_mean_signal_sess123
        

    def roi_corr_sess123(self):
        self.description = "ROI correlations per session."
        self.specific_params = get_specific_params(
            scale=False
        )
        self.analysis_fct = analyse_figs.roi_corr_sess123
        self.plot_fct = plot_figs.plot_roi_corr_sess123

    
    ### Figure 8 ###
    def dendritic_roi_tracking_example(self):
        self.description = "Dendritic ROI matching examples."
        self.specific_params = get_specific_params(
            plane="dend",
            mouse_n=6,
            tracked=True,
        )
        self.analysis_fct = analyse_figs.dendritic_roi_tracking_example
        self.plot_fct = plot_figs.plot_dendritic_roi_tracking_example
        self.warnings.append(memory_demand_warning())


    def somatic_roi_tracking_example(self):
        self.description = "Somatic ROI matching examples."
        self.specific_params = get_specific_params(
            plane="soma",
            mouse_n=4,
            tracked=True,
        )
        self.analysis_fct = analyse_figs.somatic_roi_tracking_example
        self.plot_fct = plot_figs.plot_somatic_roi_tracking_example
        self.warnings.append(memory_demand_warning())

    
    ### Figure 9 ###
    def stimulus_onset_sess123(self):
        self.description = "ROI response to stimulus onset."
        self.specific_params = get_specific_params(
            stimtype="both",
            pre=2,
            post=2,
        )
        self.analysis_fct = analyse_figs.stimulus_onset_sess123
        self.plot_fct = plot_figs.plot_stimulus_onset_sess123


    def stimulus_offset_sess123(self):
        self.description = "ROI response to stimulus offset."
        self.specific_params = get_specific_params(
            stimtype="both",
            pre=2,
            post=2,
        )
        self.analysis_fct = analyse_figs.stimulus_offset_sess123
        self.plot_fct = plot_figs.plot_stimulus_offset_sess123


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
                }
            }
        return self._figure_panel_dict

