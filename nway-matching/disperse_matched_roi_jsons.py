import argparse
import glob
import os
import re
import shutil

import pandas as pd


#############################################
def get_target_direcs(matching_file_sources, data_direc):
    """
    Returns the target directories (mouse/session specific) for each json
    """
    
    # get file names
    matching_files_local = [os.path.split(matching_file)[-1] 
        for matching_file in matching_file_sources]
    
    # identify target directories
    file_name_tokens = [matching_file_local.split("_") 
        for matching_file_local in matching_files_local]
    mouse_ids = [tokens[tokens.index("mouse") + 1] 
        for tokens in file_name_tokens]
    sess_ids = [tokens[tokens.index("session") + 1] 
        for tokens in file_name_tokens]
    
    target_direcs = [glob.glob(
        os.path.join(
            data_direc, 
            f"mouse_{mouse_ids[i]}", 
            f"ophys_session_{sess_ids[i]}", 
            f"ophys_experiment_*", 
            "processed")
            )[0] for i in range(len(file_name_tokens))]
    
    # identify target paths
    matching_file_targets = [
        os.path.join(target_direc, matching_file_local) 
        for (target_direc, matching_file_local) in 
        zip(target_direcs, matching_files_local)
    ]
    
    return matching_file_targets


#############################################
def disperse_jsons(datadir, verbose=False):

    if not os.path.exists(datadir):
        raise ValueError(f"{datadir} does not exist.")

    matching_files_direc = "nway-matched-dfs--iou-min-0.3"
    if not os.path.exists(matching_files_direc):
        raise ValueError(f"{matching_files_direc} directory is missing.")

    matching_file_sources = glob.glob(
        os.path.join(matching_files_direc, "*.json"))
    matching_file_targets = get_target_direcs(matching_file_sources, datadir)

    # copy files over
    for (file_source, file_target) in zip(
        matching_file_sources, matching_file_targets):
        
        shutil.copy(file_source, file_target)
        if verbose:
            target_direc, target_file = os.path.split(file_target)
            print(f"Copied {target_file} to {target_direc}.")


#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

        # data parameters
    parser.add_argument("--datadir", 
        help="target data directory (up to, but excluding mouse directories)")
    parser.add_argument("-v", "--verbose", action="store_true",
        help="Print copied file paths to console.")

    args = parser.parse_args()

    disperse_jsons(args.datadir, args.verbose)

    