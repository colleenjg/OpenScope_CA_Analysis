#!/usr/bin/env python

"""
sess_download_util.py

This module contains functions for downloading the dataset from the Dandi 
archive.

Dandiset 000037 is the Credit Assignment Project dandiset. It comprises data 
for 49 sessions. The asset (session file) sizes are in the following ranges:
- Basic data (with everything required for most analyses): 
    130 MB to 1.7 GB per asset
    ~25 GB total
- Basic data + stimulus template images: 
    1.5 to 3.1 GB per asset
- Basic data + stimulus template images + full imaging stack: 
    ~65 GB per asset? (to be confirmed)

URL: https://gui.dandiarchive.org/#/dandiset/000037

Authors: Colleen Gillon

Date: January, 2022

Note: this code uses python 3.7.

"""
import argparse
import logging
from pathlib import Path

import numpy as np
from dandi import dandiapi
from dandi import download as dandi_download

from util import gen_util 

gen_util.extend_sys_path(__file__, parents=2)
from util import gen_util, logger_util
from sess_util import sess_gen_util


logger = logging.getLogger(__name__)

DEFAULT_MOUSE_DF_PATH = Path("mouse_df.csv") # if running from the main directory


#############################################
def get_dandiset_asset_urls(dandiset_id="000037", version="draft", 
                            asset_sessids="all", incl_stim_templates=False, 
                            incl_full_stacks=False):
    """
    get_dandiset_asset_urls
    """
	
    client = dandiapi.DandiAPIClient()
    dandi = client.get_dandiset(dandiset_id, version)
	
    if asset_sessids != "all":
        if isinstance(asset_sessids, list):
            asset_sessids = [
                str(asset_sessid) for asset_sessid in asset_sessids
                ]
        else:
            asset_sessids = [str(asset_sessids)]

    asset_urls = []
    selected_asset_sessids = []
    for asset in dandi.get_assets():
        asset_path = Path(asset.path)
        asset_sessid = asset_path.parts[1].split("_")[1].replace("ses-", "")
        if asset_sessids != "all" and asset_sessid not in asset_sessids:
            continue

        stim_templates_included = "+image" in asset_path.parts[1]
        if stim_templates_included != incl_stim_templates:
            continue
        if incl_full_stacks:
            raise NotImplementedError(
                "Identifying assets with full imaging stacks is not "
                "implemented yet."
                )
                    
        asset_urls.append(asset.download_url)
        selected_asset_sessids.append(asset_sessid)
    
    asset_urls = [asset_urls[i] for i in np.argsort(selected_asset_sessids)]

    if len(asset_urls) == 0:
        raise RuntimeError("No dandiset assets found that meet the criteria.")
    
    return asset_urls


#############################################
def reformat_n(n):
    """
    reformat_n(n)

    Returns reformatted n argument, converting ranges to lists.

    Required args:
        - n (str): 
            number or range (e.g., "1-1", "all")
    
    Returns:
        - n (str or list): 
            number or range (e.g., [1, 2, 3], "all")
    """

    if isinstance(n, (list, int)):
        return n

    if "-" in str(n):
        vals = str(n).split("-")
        if len(vals) != 2:
            raise ValueError("If n is a range, must have format 1-3.")
        st = int(vals[0])
        end = int(vals[1]) + 1
        n = list(range(st, end))
    
    elif n not in ["any", "all"]:
        n = gen_util.list_if_not(n)

    return n


#############################################
def download_dandiset_assets(dandiset_id="000037", version="draft", output=".", 
                             incl_stim_templates=False, incl_full_stacks=False,
                             sess_ns="all", mouse_ns="all", excluded_sess=True,
                             mouse_df=DEFAULT_MOUSE_DF_PATH):

    dandiset_id = f"{int(dandiset_id):06}" # ensure correct ID formatting

    asset_sessids = "all"
    if sess_ns not in ["all", "any"] or mouse_ns not in ["all", "any"]:
        if dandiset_id != "000037":
            raise NotImplementedError(
                "Selecting assets based on session and mouse numbers is only "
                "implemented for dandiset 000037."
                )
        sess_ns = reformat_n(sess_ns)
        mouse_ns = reformat_n(mouse_ns)
        pass_fail = "all" if excluded_sess else "P"
        asset_sessids = sess_gen_util.get_sess_vals(
            mouse_df, 
            "dandi_session_id", 
            mouse_n=mouse_ns, 
            sess_n=sess_ns, 
            runtype="prod", 
            pass_fail=pass_fail, 
            incl="all", 
            sort=True
            )

    logger.info("Identifying the URLs of dandi assets to download...")
    dandiset_urls = get_dandiset_asset_urls(
        dandiset_id, 
        version=version, 
        asset_sessids=asset_sessids, 
        incl_stim_templates=incl_stim_templates, 
        incl_full_stacks=incl_full_stacks
        )

    logger.info(
        f"Downloading {len(dandiset_urls)} assets from "
        f"dandiset {dandiset_id}..."
        )

    for dandiset_url in dandiset_urls:
        dandi_download.download(dandiset_url, output, existing="refresh")


#############################################
if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument("--dandiset_id", default="000037", 
        help=("ID of the dandiset from which to download assets"))
    parser.add_argument("--version", default="draft", 
        help="version of the dandiset from which to download assets")
    parser.add_argument("--output", default=".", type=Path,
        help="where to store the dandiset files")

    # arguments applying only to dandiset 000037
    parser.add_argument("--sess_ns", default="1-3", 
        help="session numbers of assets to download (e.g., 1, 1-3 or all)")
    parser.add_argument("--mouse_ns", default="all", 
        help="mouse numbers of assets to download (e.g., 1, 1-3 or all)")
    parser.add_argument("--excluded_sess", action="store_true", 
        help=("if True, all assets (even those excluded from the paper "
            "analyses) are downloaded."))
    parser.add_argument("--mouse_df", default=DEFAULT_MOUSE_DF_PATH, type=Path, 
        help="path to mouse_csv.df, if downloading by sess_ns or mouse_ns")

    # type of asset to download
    parser.add_argument("--incl_stim_templates", action="store_true", 
        help=("if True, assets containing the stimulus templates are "
            "downloaded (~1.5 to 3.1 GB per asset for dandiset 000037)"))
    parser.add_argument("--incl_full_stacks", action="store_true", 
        help="if True, assets containing the full imaging stack are downloaded")

    args = parser.parse_args()

    download_dandiset_assets(**args.__dict__)

