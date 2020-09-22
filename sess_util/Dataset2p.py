# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:05:33 2016

Code written and shared by colleagues at the Allen Institute for Brain Science.

@author: derricw

A few tools for calculating monitor delay an ophys experiment.

"""

import os
import pickle
import json
import collections
import traceback
import logging

import h5py
import numpy as np
import matplotlib.pyplot as plt
from allensdk.brain_observatory import sync_dataset
from allensdk.brain_observatory.behavior.sync.process_sync import filter_digital

from util import gen_util, logger_util

logger = logging.getLogger(__name__)

# set a few basic parameters
ASSUMED_DELAY = 0.0351
DELAY_THRESHOLD = 0.001
FIRST_ELEMENT_INDEX = 0
SECOND_ELEMENT_INDEX = 1
SKIP_FIRST_ELEMENT = 1
SKIP_LAST_ELEMENT = -1
ROUND_PRECISION = 4
ZERO = 0
ONE = 1
TWO = 2
MIN_BOUND = .03
MAX_BOUND = .04


def calculate_stimulus_alignment(stim_time, valid_twop_vsync_fall):
    """
    calculate_stimulus_alignment(stim_time, valid_twop_vsync_fall)

    """

    logger.info("Calculating stimulus alignment.")

    # convert stimulus frames into twop frames
    stimulus_alignment = np.empty(len(stim_time))

    for index in range(len(stim_time)):
        crossings = np.nonzero(np.ediff1d(np.sign(
            valid_twop_vsync_fall - stim_time[index])) > ZERO)
        try:
            stimulus_alignment[index] = int(
                crossings[FIRST_ELEMENT_INDEX][FIRST_ELEMENT_INDEX])
        except:
            stimulus_alignment[index] = np.NaN

    return stimulus_alignment


def calculate_valid_twop_vsync_fall(sync_data, sample_frequency):
    """
    calculate_valid_twop_vsync_fall(sync_data, sample_frequency)

    """

    ####microscope acquisition frames####
    # get the falling edges of 2p
    twop_vsync_fall = sync_data.get_falling_edges("2p_vsync") / sample_frequency

    if len(twop_vsync_fall) == 0:
        raise ValueError("Error: twop_vsync_fall length is 0, possible "
                         "invalid, missing, and/or bad data")

    ophys_start = twop_vsync_fall[0]

    # only get data that is beyond the start of the experiment
    valid_twop_vsync_fall = twop_vsync_fall[
        np.where(twop_vsync_fall > ophys_start)[FIRST_ELEMENT_INDEX]]

    # skip the first element to eliminate the DAQ pulse
    return valid_twop_vsync_fall


def calculate_stim_vsync_fall(sync_data, sample_frequency):
    """
    calculate_stim_vsync_fall(sync_data, sample_frequency)

    """

    ####stimulus frames####
    # skip the first element to eliminate the DAQ pulse
    stim_vsync = sync_data.get_falling_edges("stim_vsync")[SKIP_FIRST_ELEMENT:]
    stim_vsync_fall = stim_vsync / sample_frequency

    return stim_vsync_fall


class Dataset2p(sync_dataset.Dataset):
    """
    Extends the sync.Dataset class to include some 2p-experiment-specific
        functions.  Stores a cache of derived values so subsequent calls can
        be faster.

    Args:
        path (str): path to sync file

    Example:

        >>> dset = Dataset2p("sync_data.h5")
        >>> dset.display_lag
        0.05211234
        >>> dset.stimulus_start
        31.8279931238
        >>> dset.plot_start()  # plots the start of the experiment
        >>> dset.plot_end() # plots the end fo the experiment
        >>> dset.plot_timepoint(31.8)  # plots a specific timepoint

    """

    def __init__(self, path):
        super(Dataset2p, self).__init__(path)

        self._cache = {}

    def signal_exists(self, line_name):
        """
        Checks to see if there are any events on a specified line.
        """
        if len(self.get_events_by_line(line_name)) > 0:
            return True
        else:
            return False

    @property
    def sample_freq(self):
        """
        The frequency that the sync sampled at.
        """
        return self.meta_data["ni_daq"]["counter_output_freq"]

    @property
    def display_lag(self):
        """
        The display lag in seconds.  This is the latency between the display
            buffer flip pushed by the video card and buffer actually being
            drawn on screen.
        """
        # pd0 = self.stimulus_start
        # vs0 = self.get_stim_vsyncs()[0]
        # return pd0-vs0
        stim_vsyncs = self.get_stim_vsyncs()
        transitions = stim_vsyncs[::60]
        photodiode_events = self.get_real_photodiode_events()[0:len(transitions)]
        return np.mean(photodiode_events - transitions)

    @property
    def stimulus_start(self):
        """
        The start of the visual stimulus, accounting for display lag.
        """
        return self.get_photodiode_events()[0]

    @property
    def stimulus_end(self):
        """
        The end of the visual stimulus, accounting for display lag.
        """
        vs_end = self.get_stim_vsyncs()[-1]
        return vs_end + self.display_lag

    @property
    def stimulus_duration(self):
        """
        The duration of the visual stimulus.
        """
        return self.stimulus_end - self.stimulus_start

    @property
    def twop_start(self):
        """
        The start of the two-photon acquisition.
        """
        return self.get_twop_vsyncs()[0]

    @property
    def twop_end(self):
        """
        The start of the two-photon acquisition.
        """
        return self.get_twop_vsyncs()[-1]

    @property
    def twop_duration(self):
        return self.twop_end - self.twop_start

    @property
    def video_duration(self):
        return [v[-1] - v[0] for v in self.get_video_vsyncs()]

    def get_long_stim_frames(self, threshold=0.025):
        """
        Get dropped frames for the visual stimulus using a duration threshold.

        Args:
            threshold (float): minimum duration in seconds of a "long" frame.
        """
        vsyncs = self.get_stim_vsyncs()
        vs_intervals = self.get_stim_vsync_intervals()
        drop_indices = np.where(vs_intervals > threshold)
        drop_intervals = vs_intervals[drop_indices]
        drop_times = vsyncs[drop_indices]  # maybe +1???
        return {"indices": drop_indices,
                "intervals": drop_intervals,
                "times": drop_times}

    def get_long_twop_frames(self, threshold=0.040):
        """
        Get dropped frames for the two photon using a duration threshold.

        Args:
            threshold (float): minimum duration in seconds of a "long" frame.
        """
        vsyncs = self.get_twop_vsyncs()
        vs_intervals = self.get_twop_vsync_intervals()
        drop_indices = np.where(vs_intervals > threshold)
        drop_intervals = vs_intervals[drop_indices]
        drop_times = vsyncs[drop_indices]  # maybe +1???
        return {"indices": drop_indices,
                "intervals": drop_intervals,
                "times": drop_times}

    def clear_cache(self):
        """
        Clears the cache of derived values.
        """
        self._cache = {}

    def get_photodiode_events(self):
        """
        Returns the photodiode events with the start/stop indicators and the
            window init flash stripped off.
        """
        if "pd_events" in self._cache:
            return self._cache["pd_events"]

        pd_name = "stim_photodiode"

        all_events = self.get_events_by_line(pd_name)
        pdr = self.get_rising_edges(pd_name)
        pdf = self.get_falling_edges(pd_name)

        all_events_sec = all_events / self.sample_freq
        pdr_sec = pdr / self.sample_freq
        pdf_sec = pdf / self.sample_freq

        pdf_diff = np.ediff1d(pdf_sec, to_end=0)
        pdr_diff = np.ediff1d(pdr_sec, to_end=0)

        reg_pd_falling = pdf_sec[(pdf_diff >= 1.9) & (pdf_diff <= 2.1)]

        short_pd_rising = pdr_sec[(pdr_diff >= 0.1) & (pdr_diff <= 0.5)]

        first_falling = reg_pd_falling[0]
        last_falling = reg_pd_falling[-1]

        end_indicators = short_pd_rising[short_pd_rising > last_falling]
        first_end_indicator = end_indicators[0]

        pd_events = all_events_sec[(all_events_sec >= first_falling) &
                                   (all_events_sec < first_end_indicator)]
        self._cache["pd_events"] = pd_events
        return pd_events

    def get_photodiode_anomalies(self):
        """
        Gets any anomalous photodiode events.
        """
        if "pd_anomalies" in self._cache:
            return self._cache["pd_anomalies"]

        events = self.get_photodiode_events()
        intervals = np.diff(events)
        anom_indices = np.where(intervals < 0.5)
        anom_intervals = intervals[anom_indices]
        anom_times = events[anom_indices]

        anomalies = {"indices": anom_indices,
                     "intervals": anom_intervals,
                     "times": anom_times, }

        self._cache["pd_anomalies"] = anomalies
        return anomalies

    def get_real_photodiode_events(self):
        """
        Gets the photodiode events with the anomalies removed.
        """
        events = self.get_photodiode_events()
        anomalies = self.get_photodiode_anomalies()["indices"]
        return np.delete(events, anomalies)

    def get_stim_vsyncs(self):
        """
        Returns the stimulus vsyncs in seconds, which is the falling edges of
            the "stim_vsync" signal.
        """
        if "stim_vsyncs" in self._cache:
            return self._cache["stim_vsyncs"]

        sig_name = "stim_vsync"

        svs_r = self.get_rising_edges(sig_name)
        svs_f = self.get_falling_edges(sig_name)

        svs_r_sec = svs_r / self.sample_freq
        svs_f_sec = svs_f / self.sample_freq

        # Some versions of camstim caused a spike when the DAQ is first
        # initialized.  remove it if so.
        if svs_r_sec[1] - svs_r_sec[0] > 0.2:
            vsyncs = svs_f_sec[1:]
        else:
            vsyncs = svs_f_sec

        self._cache["stim_vsyncs"] = vsyncs

        return vsyncs

    def get_stim_vsync_intervals(self):
        return np.diff(self.get_stim_vsyncs())

    def get_twop_vsync_intervals(self):
        return np.diff(self.get_twop_vsyncs())

    def get_twop_vsyncs(self):
        """
        Returns the 2p vsyncs in seconds, which is the falling edges of the
            "2p_vsync" signal.
        """
        # this one is straight-forward
        return self.get_falling_edges("2p_vsync") / self.sample_freq

    def get_video_vsyncs(self):
        """
        Returns the video monitoring system vsyncs.
        """
        vsyncs = []
        for sync_signal in ["cam1_exposure", "cam2_exposure"]:
            falling_edges = self.get_falling_edges(sync_signal, units="sec")
            rising_edges = self.get_rising_edges(sync_signal, units="sec")
            rising, falling = filter_digital(rising_edges,
                                             falling_edges,
                                             threshold=0.000001)
            vsyncs.append(falling)
        return vsyncs

    def plot_timepoint(self,
                       time_sec,
                       width_sec=3.0,
                       signals=[],
                       out_file=""):
        """
        Plots signals around a specific timepoint, with adjustable
            width.

        Args:
            time_sec (float): time to plot at in seconds
            width_sec (float): width of the time range to plot
            signals (optional[list]): list of signals to plot

        """
        if not signals:
            # defaults
            signals = ["2p_vsync", "stim_vsync", "stim_photodiode", ]

        start = time_sec - width_sec / 2
        stop = time_sec + width_sec / 2

        if out_file:
            auto_show = False
        else:
            auto_show = True

        fig = self.plot_lines(signals, start_time=start, end_time=stop,
                              auto_show=auto_show)
        if out_file:
            plt.ioff()
            plt.savefig(out_file, dpi=200)
            plt.close()

        return fig

    def plot_start(self, out_file=""):
        """
        Plots the start of the experiment.
        """
        start_time = self.stimulus_start
        return self.plot_timepoint(start_time, out_file=out_file)

    def plot_end(self, out_file=""):
        """
        Plots the end of the experiment.
        """
        end_time = self.stimulus_end
        return self.plot_timepoint(end_time, out_file=out_file)

    def plot_stim_vsync_intervals(self, out_file=""):
        """
        Plots the vsync intervals for the stimulus.
        """
        intervals = self.get_stim_vsync_intervals()
        plt.plot(intervals)
        plt.xlabel("frame number")
        plt.ylabel("duration (ms)")
        plt.title("Stimulus vsync intervals.")
        fig = plt.gcf()
        if out_file:
            plt.ioff()
            plt.savefig(out_file, dpi=200)
            plt.close()
        else:
            plt.show()
        return fig

    def plot_twop_vsync_intervals(self, out_file=""):
        """
        Plots the vsync intervals for the two-photon data.
        """
        intervals = self.get_twop_vsync_intervals()
        plt.plot(intervals)
        plt.xlabel("frame number")
        plt.ylabel("duration (ms)")
        plt.title("2p vsync intervals.")
        fig = plt.gcf()
        if out_file:
            plt.ioff()
            plt.savefig(out_file, dpi=200)
            plt.close()
        else:
            plt.show()
        return fig

    def plot_videomon_vsync_intervals(self, out_file=""):
        vsyncs = self.get_video_vsyncs()
        intervals = [np.diff(v) for v in vsyncs]
        subplots = len(intervals)
        f, axes = plt.subplots(subplots, sharex=True, sharey=True)
        if not isinstance(axes, collections.Iterable):
            axes = [axes]
        for data, ax in zip(intervals, axes):
            ax.plot(data)
        plt.xlabel("frame index")
        plt.ylabel("duration (ms)")
        if out_file:
            plt.ioff()
            plt.savefig(out_file, dpi=200)
            plt.close()
        else:
            plt.show()
        return f

    def plot_stim_frame_hist(self, out_file=""):
        """
        Plots the visual stimulus frame histogram.  This is good for
            visualizing how "clean" your frame intervals are.
        """
        intervals = self.get_stim_vsync_intervals()
        plt.hist(intervals, bins=100, range=[0.016, 0.018])
        plt.xlabel("duration (sec)")
        plt.ylabel("frames")
        fig = plt.gcf()
        if out_file:
            plt.ioff()
            plt.savefig(out_file, dpi=200)
            plt.close()
        else:
            plt.show()
        return fig

