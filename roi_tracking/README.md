# ROI tracking

This directory contains information for tracking ROIs across sessions, organized in `json` files for each session.

&nbsp;

## Dispersing ROI nway-matching jsons

If using the **original data format**, the `json` files should be dispersed into the correct subdirectories. To do this, run the following command, specifying the main target data directory (up to, but excluding mouse subdirectories):

`python disperse_matched_roi_jsons.py --datadir path/to/data`

Optionally add the `-v` switch to print each copied file path to the console.

**NOTE:** This does **not apply** to the NWB data, which is not organized hierarchically in subdirectories, and already contains most of the tracking information needed. 

