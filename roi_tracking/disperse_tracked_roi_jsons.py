import argparse
import glob
import shutil
from pathlib import Path


#############################################
def get_target_direcs(tracking_file_sources, data_direc):
    """
    get_target_direcs(tracking_file_sources, data_direc)

    Returns the target directories (mouse/session specific) for each json.

    Required args:
        - tracking_file_sources (list): source json paths

    Returns:
        - datadir (Path): target data directory
    """
    
    # get file names
    tracking_files_local = [
        tracking_file.name for tracking_file in tracking_file_sources
    ]
    
    # identify target directories
    file_name_tokens = [str(tracking_file_local).split("_") 
        for tracking_file_local in tracking_files_local]
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
    tracking_file_targets = [
        target_direc.joinpath(tracking_file_local) 
        for (target_direc, tracking_file_local) in 
        zip(target_direcs, tracking_files_local)
    ]
    
    return tracking_file_targets


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
        raise OSError(f"{datadir} does not exist.")

    tracking_files_direc = Path("nway-matched-dfs")
    if not tracking_files_direc.exists():
        raise OSError(f"{tracking_files_direc} directory is missing.")

    tracking_file_sources = sorted([Path(file_direc) 
        for file_direc in glob.glob(str(Path(tracking_files_direc, "*.json")))
    ])
    tracking_file_targets = get_target_direcs(tracking_file_sources, datadir)

    # copy files over
    for (file_source, file_target) in zip(
        tracking_file_sources, tracking_file_targets
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

    
