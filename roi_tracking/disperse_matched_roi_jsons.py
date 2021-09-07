import argparse
import glob
import shutil
from pathlib import Path


#############################################
def get_target_direcs(matching_file_sources, data_direc):
    """
    get_target_direcs(matching_file_sources, data_direc)

    Returns the target directories (mouse/session specific) for each json.

    Required args:
        - matching_file_sources (list): source json paths

    Returns:
        - datadir (Path): target data directory
    """
    
    # get file names
    matching_files_local = [
        matching_file.name for matching_file in matching_file_sources
    ]
    
    # identify target directories
    file_name_tokens = [str(matching_file_local).split("_") 
        for matching_file_local in matching_files_local]
    mouse_ids = [tokens[tokens.index("mouse") + 1] 
        for tokens in file_name_tokens]
    sess_ids = [tokens[tokens.index("session") + 1] 
        for tokens in file_name_tokens]
    
    target_direcs = [
        Path(
            glob.glob(str(Path(
                data_direc, 
                f"mouse_{mouse_ids[i]}", 
                f"ophys_session_{sess_ids[i]}", 
                f"ophys_experiment_*", 
                "processed"
                )))[0]
            ) for i in range(len(file_name_tokens))
        ]
    
    # identify target paths
    matching_file_targets = [
        target_direc.joinpath(matching_file_local) 
        for (target_direc, matching_file_local) in 
        zip(target_direcs, matching_files_local)
    ]
    
    return matching_file_targets


#############################################
def disperse_jsons(datadir, verbose=False):
    """
    disperse_jsons(datadir)

    Disperses jsons in the data directories.

    Required args:
        - datadir (Path): target data directory

    Optional args:
        - verbose (bool): if True, copying logs are printed to the console
                          default: False
    """

    datadir = Path(datadir)
    if not datadir.exists():
        raise ValueError(f"{datadir} does not exist.")

    matching_files_direc = Path("nway-matched-dfs--iou-min-0.3")
    if not matching_files_direc.exists():
        raise ValueError(f"{matching_files_direc} directory is missing.")

    matching_file_sources = [Path(file_direc) 
        for file_direc in glob.glob(str(Path(matching_files_direc, "*.json")))
    ]
    matching_file_targets = get_target_direcs(matching_file_sources, datadir)

    # copy files over
    for (file_source, file_target) in zip(
        matching_file_sources, matching_file_targets
        ):
        
        shutil.copy(file_source, file_target)
        if verbose:
            target_direc = file_target.parent
            target_file = file_target.name
            print(f"Copied {target_file} to {target_direc}.")


#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

        # data parameters
    parser.add_argument("--datadir", type=Path,
        help="target data directory (up to, but excluding mouse directories)")
    parser.add_argument("-v", "--verbose", action="store_true",
        help="Print copied file paths to console.")

    args = parser.parse_args()

    disperse_jsons(args.datadir, args.verbose)

    