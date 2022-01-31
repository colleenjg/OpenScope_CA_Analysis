# ROI tracking

This directory contains information for tracking ROIs across sessions, organized in `json` files for each session.  

&nbsp;

## Dispersing ROI nway-matching jsons 

If using the **original data format**, the `json` files should be dispersed into the correct subdirectories. To do this, run the following command, specifying the main target data directory (up to, but excluding mouse subdirectories):

`python disperse_matched_roi_jsons.py --target_dir path/to/data`

Optionally add the `-v` switch to print each copied file path to the console.

**NOTE:** This does **not apply** to the data in NWB format (downloaded from the DANDI archive) which (1) is not organized hierarchically in subdirectories, and (2) already contains most of the tracking information needed. (In cases where the info needed is _not present_ in the NWB files, the codebase will load it from the correct file in this folder.) 

