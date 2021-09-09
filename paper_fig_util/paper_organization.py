"""
paper_organization.py

This script contains functions and objects for linking analyses to the paper 
structure.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging
from pathlib import Path

from util import gen_util, logger_util
from sess_util import sess_gen_util
from paper_fig_util import corr_figs, decoding_figs, misc_figs, \
    run_pupil_figs, seq_figs, tracking_figs, usi_figs, plot_figs

logger = logging.getLogger(__name__)


PAPER_SEED = 905
DEFAULT_LOW_POWER = 1e3


#############################################
def get_all_panels(figure=2):
    """
    get_all_panels()

    Returns all panels for a figure.

    Optional args:
        - figure (int or str): figure number (e.g., 2 or 'S2')
                               default: 2
    
    Returns:
        - all_panels (list): list of panels for the figure
    """

    # create a dummy object with figure/panel combination that exists.
    dummy_fig_panel = FigurePanelAnalysis(figure=2, panel="G", datadir="")

    figure = str(figure)
    if figure not in dummy_fig_panel.figure_panel_dict.keys():
        raise ValueError("Only the following figure values are "
            f"accepted: {dummy_fig_panel.figure_panel_dict.keys()}.")
    
    all_panels = dummy_fig_panel.figure_panel_dict[figure].keys()
    
    return all_panels


### DEFINE FUNCTIONS RETURNING ERRORS AND WARNING MESSAGES
#############################################
def no_plot_fct(reason):
    raise ValueError(f"Cannot plot figure panel as it {reason}.")

def partial_plot_fct(message):
    return message

def manual_warning():
    message = ("Minor manual formatting adjustments may be missing.")
    return message

def slow_warning():
    message = ("This figure panel takes longer to produce, as it requires "
        "plotting and rasterizing a large number of traces.")
    return message

def stats_plot_fct():
    message = ("This figure panel includes statistical analyses. "
        "Analysis may take longer, and statistical symbols may not be "
        "nicely spaced out.")
    return message

def seed_warning(seed):
    message = ("Using a different seed from the one used in "
        f"the paper: {seed}. Results may differ slightly "
        "from published results. To use paper seed, run script with "
        "default seed argument, i.e., '--seed paper'.")
    return message

def power_warning():
    message = ("Reducing number of permutations/shuffles to reduce "
        "computation time. This weakens statistical power for significance "
        "testing a bit, potentially producing results that differ slightly "
        "from published results. To reproduce paper results exactly, run "
        "script with the '--full_power' argument.")
    return message



### DEFINE FUNCTION COLLECTING FIGURE/PANEL SPECIFIC PARAMETERS
#############################################
def get_specific_params(scale=True, sess_n="1-3", mouse_n="any", plane="all", 
                        line="all", stimtype="gabors", gabfr=3, gab_ori="any", 
                        pre=0, post=0.6, tails=2, idx_feature="by_exp", 
                        roi=True, run=False, pupil=False):
    """
    get_specific_params()

    Returns specific parameters for the analysis.

    Optional args:
        - scale (bool): 
            whether to use ROI scaling
            default: True
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

    bri_dir, bri_size, gabfr, gabk, gab_ori = sess_gen_util.get_params(
        stimtype, gabfr=gabfr, gab_ori=gab_ori
        )

    specific_params = {
        "scale"      : scale,
        "sess_n"     : sess_n,
        "mouse_n"    : mouse_n,
        "plane"      : plane,
        "line"       : line,
        "stimtype"   : stimtype,
        "bri_dir"    : bri_dir,
        "bri_size"   : bri_size,
        "gabfr"      : gabfr,
        "gabk"       : gabk,
        "gab_ori"    : gab_ori,
        "pre"        : pre,
        "post"       : post,
        "tails"      : tails,
        "idx_feature": idx_feature,
        "roi"        : roi,
        "run"        : run,
        "pupil"      : pupil,
    }

    return specific_params


#############################################
class FigurePanelAnalysis():
    def __init__(self, figure, panel, datadir, mouse_df_path="mouse_df.csv", 
                 output="paper_figures", full_power=False, parallel=False, 
                 seed="paper", plt_bkend=None, fontdir=None):
        """
        Initializes a FigurePanelAnalysis object.

        Sets attributes from input arguments, and runs self._set_plot_info().

        Required args:
            - figure (str): figure 
            - panel (str): panel
            - datadir (Path): data directory
        
        Optional args:
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
        
        self.figure = str(figure)
        self.panel  = str(panel).upper()

        self.datadir       = Path(datadir)
        self.mouse_df_path = Path(mouse_df_path)
        self.output        = Path(output)
        
        self.parallel  = parallel
        self.plt_bkend = plt_bkend
        self.fontdir   = fontdir

        self.warnings = [manual_warning()]
        self.full_power = full_power
        self.seed       = seed

        self._set_plot_info()
        

    def _set_power(self, n_perms_low=DEFAULT_LOW_POWER):
        """
        self._set_power()

        Sets and updates attributes related to statistical power for an 
        analysis.

        Optional args:
            - n_perms_low (bool): 
                number of permutations to use if power is not full_power

        Sets the following attributes:
            - n_perms (int): number of permutations that will be used
            - randomness (bool): whether randomness is involved in the analysis

        Updates the following attributes:
            - full_power (bool): 
                whether analysis will be run at full statistic power (same 
                number of permutations as used in the paper)
        """

        if not hasattr(self, "description"):
            raise ValueError("Must run self._set_plot_info() first.")

        self.randomness = False

        if hasattr(self, "n_perms_full"):
            self.randomness = True
            if self.full_power:
                self.n_perms = self.n_perms_full
            else:
                if self.n_perms_full > n_perms_low:
                    self.n_perms = n_perms_low
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
        
        if not hasattr(self, "randomness"):
            raise ValueError("Must run self._set_power() first.")

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
                    self.seed = gen_util.seed_all(
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

        if self.figure not in self.figure_panel_dict.keys():
            raise ValueError("Only the following figure values are "
                f"accepted: {self.figure_panel_dict.keys()}.")
        else:
            existing_panels = self.figure_panel_dict[self.figure].keys()
            if self.panel not in existing_panels:
                existing_panel_strs = ", ".join(existing_panels)
                raise ValueError(f"Panel {self.panel} is not recognized for "
                    f"figure {self.figure}. Existing panels are "
                    f"{existing_panel_strs}.")

        self.figure_panel_dict[self.figure][self.panel]()

        self._set_power()
        self._set_seed()


    def log_warnings(self):
        """
        Logs figure/panel warning messages, stored in self.warnings, to the 
        console.
        """

        if len(self.warnings):
            warn_str = "\n- " + "\n- ".join(self.warnings)
            logger.warning(warn_str, extra={"spacing": "\n"})



    ### FUNCTIONS DEFINING FIGURE/PANEL ANALYSIS ATTRIBUTES ###
    #############################################
    ### Figure 1 ###
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


    def imaging_planes(self):
        self.description = "Example projections of 2p imaging planes."
        self.specific_params = None
        self.analysis_fct = None
        self.plot_fct = None
        no_plot_fct(reason="was drawn manually from images")


    def imaging_plane_schematic(self):
        self.description = ("Schematic illustration of the four imaging "
            "planes.")
        self.specific_params = None
        self.analysis_fct = None
        self.plot_fct = None
        no_plot_fct(reason="is a schematic illustration")


    def roi_tracking(self):
        self.description = "Example ROI tracking overlays."
        self.specific_params = get_specific_params() # most values are not used
        self.analysis_fct = tracking_figs.roi_tracking
        self.plot_fct = plot_figs.plot_roi_tracking
        self.warnings.append(
            partial_plot_fct(
                message=("Only overlays will be generated, without "
                         "additional manual formatting.")
            )
        )


    ### Figure 2 ###
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


    def gabor_example_roi_usis(self):
        self.description = "Example ROI responses to Gabor sequences."
        self.specific_params = get_specific_params(
            sess_n=1,
            pre=[0, 0.9],
            line="L23",
        )
        self.n_perms_full = 1e4
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
        self.n_perms_full = 1e4
        self.analysis_fct = usi_figs.gabor_example_roi_usi_sig
        self.plot_fct = plot_figs.plot_gabor_example_roi_usi_sig


    def gabor_roi_usi_distr(self):
        self.description = "Distributions of Gabor USI percentiles."
        self.specific_params = get_specific_params(
            sess_n=1,
        )
        self.n_perms_full = 1e4
        self.analysis_fct = usi_figs.gabor_roi_usi_distr
        self.plot_fct = plot_figs.plot_gabor_roi_usi_distr


    def gabor_roi_usi_sig(self):
        self.description = "Percentages of significant Gabor USIs."
        self.specific_params = get_specific_params(
            sess_n=1,
        )
        self.n_perms_full = 1e4
        self.analysis_fct = usi_figs.gabor_roi_usi_sig
        self.plot_fct = plot_figs.plot_gabor_roi_usi_sig
        self.warnings.append(stats_plot_fct())


    def gabor_roi_usi_sig_common_oris(self):
        self.description = ("Percentages of significant Gabor USIs for "
            "sequences with orientations common to D/U.")
        self.specific_params = get_specific_params(
            sess_n=1,
        )
        self.n_perms_full = 1e4
        self.analysis_fct = usi_figs.gabor_roi_usi_sig_common_oris
        self.plot_fct = plot_figs.plot_gabor_roi_usi_sig_common_oris
        self.warnings.append(stats_plot_fct())


    def pupil_run_responses(self):
        self.description = "Running and pupil responses to Gabor sequences."
        self.specific_params = get_specific_params(
            sess_n=1,
            pre=0.9,
            roi=False,
            run=True,
            pupil=True,
        )
        self.analysis_fct = run_pupil_figs.pupil_run_responses
        self.plot_fct = plot_figs.plot_pupil_run_responses
        self.warnings.append(
            partial_plot_fct(
                message="Running and pupil images will be missing."
            )
        )


    def pupil_run_diffs(self):
        self.description = ("Trial differences in running and pupil "
            "responses U-G vs D-G Gabor sequences.")
        self.specific_params = get_specific_params(
            sess_n=1,
            tails="hi",
            roi=False,
            run=True,
            pupil=True,
        )
        self.n_perms_full = None
        self.analysis_fct = run_pupil_figs.pupil_run_diffs
        self.plot_fct = plot_figs.plot_pupil_run_diffs


    ### Figure 3 ###
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
        self.warnings.append(stats_plot_fct())
        

    def gabor_rel_resp_sess123(self):
        self.description = ("ROI responses to regular and unexpected "
            "Gabor sequences, relative to session 1.")
        self.specific_params = get_specific_params(
            pre=0.9,
        )
        self.n_perms_full = 1e5
        self.analysis_fct = seq_figs.gabor_rel_resp_sess123
        self.plot_fct = plot_figs.plot_gabor_rel_resp_sess123
        self.warnings.append(stats_plot_fct())


    def gabor_tracked_roi_usis_sess123(self):
        self.description = "Tracked ROI Gabor USIs across sessions."
        self.specific_params = get_specific_params()
        self.n_perms_full = 1e5
        self.analysis_fct = usi_figs.gabor_tracked_roi_usis_sess123
        self.plot_fct = plot_figs.plot_gabor_tracked_roi_usis_sess123
        

    def gabor_tracked_roi_usi_means_sess123(self):
        self.description = ("Absolute means of tracked ROI Gabor USIs "
            "across sessions.")
        self.specific_params = get_specific_params()
        self.n_perms_full = 1e5
        self.analysis_fct = usi_figs.gabor_tracked_roi_usi_means_sess123
        self.plot_fct = plot_figs.plot_gabor_tracked_roi_usi_means_sess123
        self.warnings.append(stats_plot_fct())


    ### Figure 4 ###
    def gabor_decoding_sess123(self):
        self.description = ("Mean Gabor orientation decoder performances "
            "across sessions.")
        self.specific_params = get_specific_params(
            pre=0,
            post=0.6
        )
        self.n_perms_full = 1e5
        self.analysis_fct = decoding_figs.gabor_decoding_sess123
        self.plot_fct = plot_figs.plot_gabor_decoding_sess123
        self.warnings.append(stats_plot_fct())


    def gabor_norm_res_corr_example(self):
        self.description = ("Example normalized residual Gabor USI correlation "
            "between session 1 and 2.")
        self.specific_params = get_specific_params(
           line="L23",
           plane="soma", 
        )
        self.n_perms_full = 1e5
        self.analysis_fct = corr_figs.gabor_norm_res_corr_example
        self.plot_fct = plot_figs.plot_gabor_norm_res_corr_example


    def gabor_norm_res_corrs_sess123_comps(self):
        self.description = ("Normalized residual Gabor USI correlations "
            "between sessions.")
        self.specific_params = get_specific_params()
        self.n_perms_full = 1e5
        self.analysis_fct = corr_figs.gabor_norm_res_corrs_sess123_comps
        self.plot_fct = plot_figs.plot_gabor_norm_res_corrs_sess123_comps
        

    def model_illustration(self):
        self.description = ("Schematic illustration of a conceptual model "
            "based on the data.")
        self.specific_params = None
        self.analysis_fct = None
        self.plot_fct = None
        no_plot_fct(reason="is a schematic illustration")
        

    ### Figure S1 ###
    def roi_overlays_sess123(self):
        self.description = "Example ROI tracking overlays (large)."
        self.specific_params = get_specific_params() # most values are not used
        self.analysis_fct = tracking_figs.roi_overlays_sess123
        self.plot_fct = plot_figs.plot_roi_overlays_sess123
        self.warnings.append(
            partial_plot_fct(
                message=("Only overlays will be generated, without "
                         "additional manual formatting.")
            )
        )


    def roi_overlays_sess123_enlarged(self):
        self.description = "Example ROI tracking overlay close-ups (large)."
        self.specific_params = None
        self.analysis_fct = None
        self.plot_fct = None
        no_plot_fct(reason="was enlarged manually from overlays")


    ### Figure S2 ###
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
        

    ### Figure S3 ###
    def stimulus_onset_sess123(self):
        self.description = "ROI response to stimulus onset."
        self.specific_params = get_specific_params(
            stimtype="both",
            pre=2,
            post=2,
        )
        self.analysis_fct = seq_figs.stimulus_onset_sess123
        self.plot_fct = plot_figs.plot_stimulus_onset_sess123
        

    def gabor_ex_roi_responses_sess1(self):
        self.description = "Example ROI responses to each Gabor sequence."
        self.specific_params = get_specific_params(
            sess_n=1,
            pre=0.9,
            post=0.6,
            )
        self.analysis_fct = seq_figs.gabor_ex_roi_responses_sess1
        self.plot_fct = plot_figs.plot_gabor_ex_roi_responses_sess1
        self.warnings.append(slow_warning())

    ### Figure S4 ###
    def gabor_roi_usi_sig_by_mouse(self):
        self.description = ("Percentages of significant Gabor USIs for "
            "each mouse.")
        self.specific_params = get_specific_params(
            sess_n=1,
        )
        self.n_perms_full = 1e5
        self.analysis_fct = usi_figs.gabor_roi_usi_sig_by_mouse
        self.plot_fct = plot_figs.plot_gabor_roi_usi_sig_by_mouse
        self.warnings.append(stats_plot_fct())


    def gabor_rel_resp_tracked_rois_sess123(self):
        self.description = ("Tracked ROI responses to regular and unexpected "
            "Gabor sequences, relative to session 1.")
        self.specific_params = get_specific_params(
            pre=0.9,
        )
        self.n_perms_full = 1e5
        self.analysis_fct = seq_figs.gabor_rel_resp_tracked_rois_sess123
        self.plot_fct = plot_figs.plot_gabor_rel_resp_tracked_rois_sess123
        self.warnings.append(stats_plot_fct())


    def gabor_tracked_roi_means_sess123_by_mouse(self):
        self.description = ("Absolute means of tracked ROI Gabor USIs "
            "across sessions for each mouse.")
        self.specific_params = get_specific_params()
        self.n_perms_full = 1e5
        self.analysis_fct = usi_figs.gabor_tracked_roi_means_sess123_by_mouse
        self.plot_fct = plot_figs.plot_gabor_tracked_roi_means_sess123_by_mouse
        

    ### Figure S5 ###
    def visual_flow_stimulus(self):
        self.description = "Visual flow stimulus."
        self.specific_params = None
        self.analysis_fct = None
        self.plot_fct = None
        no_plot_fct(reason="was drawn manually from images")


    def visual_flow_sequences_sess123(self):
        self.description = "ROI responses to visual flow sequences."
        self.specific_params = get_specific_params(
            stimtype="bricks",
            pre=2,
            post=2,
        )
        self.analysis_fct = seq_figs.visual_flow_sequences_sess123
        self.plot_fct = plot_figs.plot_visual_flow_sequences_sess123
        

    def visual_flow_diffs_sess123(self):
        self.description = ("Differences in ROI responses to unexpected "
            "and expected visual flow sequences.")
        self.specific_params = get_specific_params(
            stimtype="bricks",
            pre=2,
            post=2,
        )
        self.n_perms_full = 1e4
        self.analysis_fct = seq_figs.visual_flow_diffs_sess123
        self.plot_fct = plot_figs.plot_visual_flow_diffs_sess123
        

    ### Figure S6 ###
    def visual_flow_rel_resp_sess123(self):
        self.description = ("ROI responses to expected and unexpected "
            "visual flow sequences, relative to session 1.")
        self.specific_params = get_specific_params(
            stimtype="bricks",
            pre=2,
            post=2,
        )
        self.n_perms_full = 1e5
        self.analysis_fct = seq_figs.visual_flow_rel_resp_sess123
        self.plot_fct = plot_figs.plot_visual_flow_rel_resp_sess123
        self.warnings.append(stats_plot_fct())


    def rel_resp_stimulus_comp_sess1v3(self):
        self.description = ("Change in ROI responses to unexpected sequences "
            "for the Gabor vs visual flow stimulus.")
        self.specific_params = get_specific_params(
            stimtype="both",
            pre=[0, 2],
            post=[0.6, 2],
        ) 
        self.n_perms_full = 1e5
        self.analysis_fct = seq_figs.rel_resp_stimulus_comp_sess1v3
        self.plot_fct = plot_figs.plot_rel_resp_stimulus_comp_sess1v3
        self.warnings.append(stats_plot_fct())


    def visual_flow_tracked_roi_usis_sess123(self):
        self.description = "Tracked ROI visual flow USIs across sessions."
        self.specific_params = get_specific_params(
            stimtype="bricks",
            pre=2,
            post=2,
        )
        self.n_perms_full = 1e5
        self.analysis_fct = usi_figs.visual_flow_tracked_roi_usis_sess123
        self.plot_fct = plot_figs.plot_visual_flow_tracked_roi_usis_sess123
        

    def visual_flow_tracked_roi_usi_means_sess123_by_mouse(self):
        self.description = ("Absolute means of tracked ROI visual flow USIs "
            "across sessions.")
        self.specific_params = get_specific_params(
            stimtype="bricks",
            pre=2,
            post=2,
        )
        self.n_perms_full = 1e5
        self.analysis_fct = usi_figs.visual_flow_tracked_roi_usi_means_sess123_by_mouse
        self.plot_fct = plot_figs.plot_visual_flow_tracked_roi_usi_means_sess123_by_mouse
        

    def tracked_roi_usis_stimulus_comp_sess1v3(self):
        self.description = ("Change in tracked ROI USIs for the Gabor vs "
            "visual flow stimulus.")
        self.specific_params = get_specific_params(
            stimtype="both",
            pre=[0, 2],
            post=[0.6, 2],
        ) 
        self.n_perms_full = 1e5
        self.analysis_fct = usi_figs.tracked_roi_usis_stimulus_comp_sess1v3
        self.plot_fct = plot_figs.plot_tracked_roi_usis_stimulus_comp_sess1v3
        

    def visual_flow_norm_res_corrs_sess123_comps(self):
        self.description = ("Normalized residual visual flow USI correlations "
            "between sessions.")
        self.specific_params = get_specific_params(
            stimtype="bricks",
            pre=2,
            post=2,
        )
        self.n_perms_full = 1e5
        self.analysis_fct = corr_figs.visual_flow_norm_res_corrs_sess123_comps
        self.plot_fct = plot_figs.plot_visual_flow_norm_res_corrs_sess123_comps
        

    ### Figure S7 ###
    def dendritic_roi_tracking_example(self):
        self.description = "Dendritic ROI matching examples."
        self.specific_params = get_specific_params(
            plane="dend"
        ) # most values are not used
        self.analysis_fct = tracking_figs.dendritic_roi_tracking_example
        self.plot_fct = plot_figs.plot_dendritic_roi_tracking_example
        self.warnings.append(
            partial_plot_fct(
                message=("Only overlays will be generated, without "
                         "additional manual formatting.")
            )
        )


    def somatic_roi_tracking_example(self):
        self.description = "Somatic ROI matching examples."
        self.specific_params = get_specific_params(
            plane="soma"
        ) # most values are not used
        self.analysis_fct = tracking_figs.somatic_roi_tracking_example
        self.plot_fct = plot_figs.plot_somatic_roi_tracking_example
        self.warnings.append(
            partial_plot_fct(
                message=("Only overlays will be generated, without "
                         "additional manual formatting.")
            )
        )


    @property
    def figure_panel_dict(self):
        """
        Sets and returns self._figure_panel_dict attribute, which maps figure 
        panels to the correct function to define analysis attributes. 
        """
        if not hasattr(self, "_figure_panel_dict"):
            self._figure_panel_dict = {
                "1": {
                    "A": self.structure_schematic,
                    "B": self.imaging_schematic,
                    "C": self.imaging_planes,
                    "D": self.imaging_plane_schematic,
                    "E": self.roi_tracking,
                    },
                "2": {
                    "A": self.gabor_sequences,
                    "B": self.experimental_timeline,
                    "C": self.gabor_example_roi_usis,
                    "D": self.gabor_example_roi_usi_sig,
                    "E": self.gabor_roi_usi_distr,
                    "F": self.gabor_roi_usi_sig,
                    "G": self.gabor_roi_usi_sig_common_oris,
                    "H": self.pupil_run_responses,
                    "I": self.pupil_run_diffs,
                    },
                "3": {
                    "A": self.gabor_sequences_sess123,
                    "B": self.gabor_sequence_diffs_sess123,
                    "C": self.gabor_rel_resp_sess123,
                    "D": self.gabor_tracked_roi_usis_sess123,
                    "E": self.gabor_tracked_roi_usi_means_sess123,
                    },
                "4": {
                    "A": self.gabor_decoding_sess123,
                    "B": self.gabor_norm_res_corr_example,
                    "C": self.gabor_norm_res_corrs_sess123_comps,
                    "D": self.model_illustration,
                    },
                "S1": {
                    "A": self.roi_overlays_sess123,
                    "B": self.roi_overlays_sess123_enlarged
                    },
                "S2": {
                    "A": self.snrs_sess123,
                    "B": self.mean_signal_sess123,
                    "C": self.nrois_sess123,
                    },
                "S3": {
                    "A": self.stimulus_onset_sess123,
                    "B": self.gabor_ex_roi_responses_sess1,
                    },
                "S4": {
                    "A": self.gabor_roi_usi_sig_by_mouse,
                    "B": self.gabor_rel_resp_tracked_rois_sess123,
                    "C": self.gabor_tracked_roi_means_sess123_by_mouse,
                    },
                "S5": {
                    "A": self.visual_flow_stimulus,
                    "B": self.visual_flow_sequences_sess123,
                    "C": self.visual_flow_diffs_sess123,
                    },
                "S6":  {
                    "A": self.visual_flow_rel_resp_sess123,
                    "B": self.rel_resp_stimulus_comp_sess1v3,
                    "C": self.visual_flow_tracked_roi_usis_sess123,
                    "D": self.visual_flow_tracked_roi_usi_means_sess123_by_mouse,
                    "E": self.tracked_roi_usis_stimulus_comp_sess1v3,
                    "F": self.visual_flow_norm_res_corrs_sess123_comps,
                    },
                "S7":  {
                    "A": self.dendritic_roi_tracking_example,
                    "B": self.somatic_roi_tracking_example,
                    }
                }
        return self._figure_panel_dict

