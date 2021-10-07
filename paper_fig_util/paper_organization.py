"""
paper_organization.py

This script contains functions and objects for linking analyses to the paper 
structure.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging
import warnings

from paper_fig_util import corr_figs, misc_figs, roi_figs, run_pupil_figs, \
                           seq_figs, usi_figs

logger = logging.getLogger(__name__)

PAPER_SEED = 905

### DEFINE FUNCTION ALERTS
def no_plot_fct(reason):
    raise ValueError(f"Cannot plot figure panel as it {reason}.")

def partial_plot_fct(message):
    warnings.warn(message)

def stats_plot_fct():
    warnings.warn("This figure panel includes statistical analyses. "
        "Analysis may take longer, and statistical symbols may not be "
        "nicely spaced out.")


def get_specific_params(scale=True, sess_n="1-3", plane="all", line="all", 
                          stimtype="gabors", gabfr=3, gab_ori="all", pre=0, 
                          post=0.6):

    specific_params = {
        "scale": scale,
        "sess_n": sess_n,
        "plane": plane,
        "line": line,
        "stimtype": stimtype,
        "gabfr": gabfr,
        "gab_ori": gab_ori,
        "pre": pre,
        "post": post,
    }

    return specific_params


class FigurePanelAnalysis():
    def __init__(self, figure, panel, datadir, mouse_df_path="mouse_df.csv", 
                 output="figure_panels", full_power=False, parallel=False, 
                 rerun_local=False, seed="paper", plt_bkend=None, fontdir=None):
        
        self.figure = str(figure)
        self.panel  = str(panel).upper()

        logger.info(f"Fig. {self.figure}{self.panel}.")

        self.datadir       = datadir
        self.mouse_df_path = mouse_df_path
        self.output        = output
        
        self.full_power  = full_power 
        if seed == "paper":
            self.seed = PAPER_SEED
        else:
            self.seed = int(seed)
        self.rerun_local = rerun_local
        
        self.parallel  = parallel
        self.plt_bkend = plt_bkend
        self.fontdir   = fontdir

    def get_plot_info(self):
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

        specific_params, plot_fct = \
            self.figure_panel_dict[self.figure][self.panel]

        return specific_params, plot_fct

    ### DEFINE FUNCTIONS
    ### Figure 1
    def structure_schematic(self):
        no_plot_fct(reason="is a schematic illustration")
        return

    def imaging_schematic(self):
        no_plot_fct(reason="is a schematic illustration")
        return

    def imaging_planes(self):
        no_plot_fct(reason="was drawn manually with images")
        return

    def imaging_plane_schematic(self):
        no_plot_fct(reason="is a schematic illustration")
        return

    def roi_tracking(self):
        partial_plot_fct(message=("Only overlays will be generated, without "
            "additional formatting."))
        specific_params = get_specific_params() # most values are not used

        return specific_params, roi_figs.roi_tracking

    ### Figure 2
    def gabor_sequences(self):
        no_plot_fct(reason="was drawn manually with images")
        return

    def experimental_timeline(self):
        no_plot_fct(reason="was drawn manually")
        return

    def gabor_example_roi_usis(self):
        partial_plot_fct(message="Some formatting adjustments will be missing.")
        specific_params = get_specific_params(
            sess_n=1,
            line="L23",
            pre=0.9,
        )
        return specific_params, usi_figs.gabor_example_roi_usis

    def gabor_example_roi_usi_sig(self):
        partial_plot_fct(message="Some formatting adjustments will be missing.")
        specific_params = get_specific_params(
            sess_n=1,
            line="L23",
            plane="soma",
        )
        return specific_params, usi_figs.gabor_example_roi_usi_sig

    def gabor_roi_usi_distr(self):
        specific_params = get_specific_params(
            sess_n=1,
        )
        return specific_params, usi_figs.gabor_roi_usi_distr

    def gabor_roi_usi_sig(self):
        stats_plot_fct()
        specific_params = get_specific_params(
            sess_n=1,
        )
        return specific_params, usi_figs.gabor_roi_usi_sig

    def gabor_roi_usi_sig_matched_oris(self):
        stats_plot_fct()
        specific_params = get_specific_params(
            sess_n=1,
            gab_ori=[0, 90],
        )
        return specific_params, usi_figs.gabor_roi_usi_sig_matched_oris

    def pupil_run_responses(self):
        partial_plot_fct(message="Pupil and running images will be missing.")
        specific_params = get_specific_params(
            sess_n=1,
            pre=0.9,
        )
        return specific_params, run_pupil_figs.pupil_run_responses

    def pupil_run_diffs(self):
        specific_params = get_specific_params(
            sess_n=1,
        )
        return specific_params, run_pupil_figs.pupil_run_diffs

    ### Figure 3
    def gabor_sequences_sess123(self):
        specific_params = get_specific_params(
            pre=0.9,
        )
        return specific_params, seq_figs.gabor_sequences_sess123
        
    def gabor_sequence_diffs_sess123(self):
        stats_plot_fct()
        specific_params = get_specific_params(
            pre=0.9,
        )
        return specific_params, seq_figs.gabor_sequence_diffs_sess123

    def gabor_rel_resp_sess123(self):
        stats_plot_fct()
        specific_params = get_specific_params(
            pre=0.9,
        )
        return specific_params, seq_figs.gabor_rel_resp_sess123

    def gabor_tracked_roi_usis_sess123(self):
        specific_params = get_specific_params()
        return specific_params, usi_figs.gabor_tracked_roi_usis_sess123

    def gabor_tracked_roi_usi_means_sess123(self):
        stats_plot_fct()
        specific_params = get_specific_params()
        return specific_params, usi_figs.gabor_tracked_roi_usi_means_sess123

    ### Figure 4
    def gabor_decoding_sess123(self):
        stats_plot_fct()
        specific_params = get_specific_params()
        return specific_params, misc_figs.gabor_decoding_sess123

    def gabor_norm_res_corr_example(self):
        partial_plot_fct(message="Some formatting adjustments will be missing.")
        specific_params = get_specific_params(
           line="L23",
           plane="soma", 
        )
        return specific_params, corr_figs.gabor_norm_res_corr_example

    def gabor_norm_res_corrs_sess123_comps(self):
        specific_params = get_specific_params()
        return specific_params, corr_figs.gabor_norm_res_corrs_sess123_comps

    def model_illustration(self):
        no_plot_fct(reason="is a schematic illustration")
        return
        
    ### Figure S1
    def roi_overlays_sess123(self):
        partial_plot_fct(message=("Only overlays will be generated, without "
            "additional formatting."))
        specific_params = get_specific_params() # most values are not used
        return specific_params, roi_figs.roi_overlays_sess123

    def roi_overlays_sess123_enlarged(self):
        no_plot_fct(reason="was enlarged manually")
        return

    ### Figure S2
    def snrs_sess123(self):
        specific_params = get_specific_params()
        return specific_params, misc_figs.snrs_sess123

    def mean_signal_sess123(self):
        specific_params = get_specific_params()
        return specific_params, misc_figs.mean_signal_sess123

    def n_rois_sess123(self):
        specific_params = get_specific_params()
        return specific_params, misc_figs.n_rois_sess123

    ### Figure S3
    def stimulus_onset_sess123(self):
        specific_params = get_specific_params()
        return specific_params, seq_figs.stimulus_onset_sess123

    def gabor_example_roi_responses_sess1(self):
        specific_params = get_specific_params()
        return specific_params, seq_figs.gabor_example_roi_responses_sess1

    ### Figure S4
    def gabor_roi_usi_sig_by_mouse(self):
        stats_plot_fct()
        specific_params = get_specific_params(
            sess_n=1,
        )
        return specific_params, usi_figs.gabor_roi_usi_sig_by_mouse

    def gabor_rel_resp_tracked_rois_sess123(self):
        stats_plot_fct()
        specific_params = get_specific_params(
            pre=0.9,
        )
        return specific_params, seq_figs.gabor_rel_resp_tracked_rois_sess123

    def gabor_tracked_roi_means_sess123_by_mouse(self):
        specific_params = get_specific_params()
        return specific_params, usi_figs.gabor_tracked_roi_means_sess123_by_mouse

    ### Figure S5
    def visual_flow_stimulus(self):
        no_plot_fct(reason="was drawn manually with images")
        return

    def visual_flow_sequences_sess123(self):
        specific_params = get_specific_params(
            stimtype="bricks",
            pre=2,
            post=2,
        )
        return specific_params, seq_figs.visual_flow_sequences_sess123

    def visual_flow_diffs_sess123(self):
        specific_params = get_specific_params(
            stimtype="bricks",
            pre=2,
            post=2,
        )
        return specific_params, seq_figs.visual_flow_diffs_sess123

    ### Figure S6
    def visual_flow_rel_resp_sess123(self):
        stats_plot_fct()
        specific_params = get_specific_params(
            stimtype="bricks",
            pre=2,
            post=2,
        )
        return specific_params, seq_figs.visual_flow_rel_resp_sess123

    def rel_resp_stimulus_comp_sess1v3(self):
        stats_plot_fct()
        specific_params = get_specific_params(
            stimtype="both",
        ) # pre/post times are fixed for this analysis
        return specific_params, seq_figs.rel_resp_stimulus_comp_sess1v3

    def visual_flow_tracked_roi_usis_sess123(self):
        specific_params = get_specific_params(
            stimtype="bricks",
            pre=2,
            post=2,
        )
        return specific_params, usi_figs.visual_flow_tracked_roi_usis_sess123

    def visual_flow_tracked_roi_usi_means_sess123_by_mouse(self):
        specific_params = get_specific_params(
            stimtype="bricks",
            pre=2,
            post=2,
        )
        return specific_params, usi_figs.visual_flow_tracked_roi_usi_means_sess123_by_mouse

    def tracked_roi_usis_stimulus_comp_sess1v3(self):
        specific_params = get_specific_params(
            stimtype="both",
        ) # pre/post times are fixed for this analysis
        return specific_params, usi_figs.tracked_roi_usis_stimulus_comp_sess1v3

    def visual_flow_norm_res_corrs_sess123_comps(self):
        specific_params = get_specific_params(
            stimtype="bricks",
            pre=2,
            post=2,
        )
        return specific_params, corr_figs.visual_flow_norm_res_corrs_sess123_comps

    ### Figure S7
    def dendritic_roi_tracking_example(self):
        partial_plot_fct(message=("Only overlays will be generated, without "
            "additional formatting."))
        specific_params = get_specific_params(
            plane="dend"
        ) # most values are not used
        return specific_params, roi_figs.dendritic_roi_tracking_example

    def somatic_roi_tracking_example(self):
        partial_plot_fct(message=("Only overlays will be generated, without "
            "additional formatting."))
        specific_params = get_specific_params(
            plane="soma"
        ) # most values are not used
        return specific_params, roi_figs.somatic_roi_tracking_example

    @property
    def figure_panel_dict(self):
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
                    "G": self.gabor_roi_usi_sig_matched_oris,
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
                    "C": self.n_rois_sess123,
                    },
                "S3": {
                    "A": self.stimulus_onset_sess123,
                    "B": self.gabor_example_roi_responses_sess1,
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

