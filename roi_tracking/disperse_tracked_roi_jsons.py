import argparse
import glob
import shutil
from pathlib import Path


#############################################
def get_target_path(tracking_file_source, target_dir):
    """
    get_target_path(tracking_file_source, target_dir)

    Returns the target path (mouse/session specific) for a source json.

    Required args:
        - tracking_file_source (Path): source json path
        - target_dir (Path)          : main target data directory
                                       (up to, but excluding mouse directories)

    Returns:
        - tracking_file_target (Path): target json path
    """
    
    # identify target directories
    source_file_name = Path(tracking_file_source).name

    file_name_tokens = str(source_file_name).split("_") 
    mouse_id = file_name_tokens[file_name_tokens.index("mouse") + 1] 
    sess_id = file_name_tokens[file_name_tokens.index("session") + 1]
    
    target_direc_pattern = Path(
        target_dir, 
        f"mouse_{mouse_id}", 
        f"ophys_session_{sess_id}", 
        f"ophys_experiment_*", 
        "processed"
        )

    target_direc = glob.glob(str(target_direc_pattern))
    
    if len(target_direc) == 0:
        raise OSError(
            f"No directory found with pattern: {target_direc_pattern}."
            )
    elif len(target_direc) > 1:
        target_matches = ", ".join([str(match) for match in target_direc])
        raise NotImplementedError(
            f"Multiple directories found with pattern: {target_direc_pattern}, "
            f"namely {target_matches}"
            )
    else:
        target_direc = target_direc[0]

    # set target path
    tracking_file_target = Path(target_direc, source_file_name)
    
    return tracking_file_target


#############################################
def disperse_jsons(target_dir, verbose=False):
    """
    disperse_jsons(target_dir)

    Disperses jsons in the target data directories.

    Required args:
        - target_dir (Path): main target data directory
                             (up to, but excluding mouse directories)

    Optional args:
        - verbose (bool): if True, copying logs are printed to the console
                          default: False
    """

    target_dir = Path(target_dir)
    if not target_dir.is_dir():
        raise OSError(f"{target_dir} is not a directory.")

    tracking_files_direc = Path("nway-matched-dfs")
    if not tracking_files_direc.is_dir():
        raise OSError(f"{tracking_files_direc} is not a directory.")

    tracking_file_sources = sorted([Path(file_direc) 
        for file_direc in glob.glob(str(Path(tracking_files_direc, "*.json")))
    ])

    # copy files over
    for file_source in tracking_file_sources:
        file_target = get_target_path(file_source, target_dir)
        shutil.copy(file_source, file_target)
        if verbose:
            source_file = file_source.name
            target_direc = file_target.parent
            print(f"Copied {source_file} to {target_direc}.")


#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

        # data parameters
    parser.add_argument("--target_dir", type=Path,
        help=("main target data directory (up to, but excluding mouse "
            "directories)"))
    parser.add_argument("-v", "--verbose", action="store_true",
        help="Print copied file paths to console.")

    args = parser.parse_args()

    disperse_jsons(args.target_dir, args.verbose)

    
