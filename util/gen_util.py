"""
gen_util.py

This module contains general purpose functions.

Authors: Colleen Gillon

Date: February 2023

Note: this code was aggregated from https://github.com/colleenjg/util.

"""

import multiprocessing
from pathlib import Path
import os
import time

from joblib import Parallel, delayed
import numpy as np


from util import logger_util

logger = logger_util.get_module_logger(name=__name__)



#############################################
class TimeIt():
    """
    Context manager for timing a function, and logging duration to the logger
    (in HHh MMmin SSsec).
    """

    def __init__(self):
        return

    def __enter__(self):
        self.start = time.perf_counter()


    def __exit__(self, exc_type, exc_value, exc_traceback):

        end = time.perf_counter()
        duration = end - self.start # in seconds

        rem_duration = duration
        hours, mins = 0, 0
        duration_str = ""
        fail_str = " (failed)" if exc_type else ""

        secs_per_hour = 60 * 60
        if duration > secs_per_hour:
            hours = int(rem_duration // secs_per_hour)
            rem_duration = rem_duration - hours * secs_per_hour
            duration_str = f"{duration_str}{hours}h "
        
        secs_per_min = 60
        if duration > secs_per_min:
            mins = int(rem_duration // secs_per_min)
            rem_duration = rem_duration - mins * secs_per_min
            duration_str = f"{duration_str}{mins}m "
        
        secs = rem_duration
        duration_str = f"{duration_str}{secs:.2f}s"

        logger.info(f"Duration: {duration_str}{fail_str}")
        

#############################################
def accepted_values_error(varname, wrong_val, accept_vals):
    """
    accepted_values_error(varname, wrong_val, accept_vals)

    Raises a value error with a message indicating the variable name,
    accepted values for that variable and wrong value stored in the variable.

    Required args:
        - varname (str)     : name of the variable
        - wrong_val (item)  : value stored in the variable
        - accept_vals (list): list of accepted values for the variable
    """

    val_str = ", ".join([f"'{x}'" for x in accept_vals])
    error_message = (f"'{varname}' value '{wrong_val}' unsupported. Must be in "
        f"{val_str}.")
    raise ValueError(error_message)
    
    
#############################################
def add_ext(filename, filetype="pickle"):
    """
    add_ext(filename)

    Returns a file name with extension added if there wasn't already an
    extension. Only adds pickle, json or csv extensions.
 
    Required args:
        - filename (Path): name of file, can include the whole directory name
                           and extension
    
    Optional args:
        - filetype (str): type of file (pickle, pkl, json, png, csv, svg, jpg).
                          Overridden if extension already in filename.
                          Can include ""
                          default: "pickle"

    Returns:
        - filename (Path): file name, including extension
        - ext (str)      : extension, including ""
    """

    filename = Path(filename)
    ext = filename.suffix

    filetype = filetype.replace(".", "")

    if ext == "":
        filetypes = ["pkl", "pickle", "json", "csv", "png", "svg", "jpg"]
        file_exts  = [".pkl", ".pkl", ".json", ".csv", ".png", ".svg", ".jpg"]
        if filetype not in filetypes:
            accepted_values_error("filetype", filetype, filetypes)
        ext = file_exts[filetypes.index(filetype)]
        filename = filename.with_suffix(ext)

    return filename, ext
    

#############################################
def get_unique_path(savename, fulldir=".", ext=None):
    """
    get_unique_path(savename)

    Returns a unique version of savename by adding numbers if a file by the 
    same name already exists. 

    Required args:
        - savename (Path): name under which to save info, can include the 
                           whole directory name and extension
   
    Optional args:
        - fulldir (Path): directory to append savename to
                          default: "."
        - ext (str)     : extension to use which, if provided, overrides any
                          extension in savename
                          default: None
    
    Returns:
        - fullname (Path): savename with full directory and extension, modified 
                           with a number if needed
    """

    savename = Path(savename)
    if ext is None:
        ext = savename.suffix
        savename = Path(savename.parent, savename.stem)
    elif "." not in ext:
        ext = f".{ext}"

    fullname = Path(fulldir, savename).with_suffix(ext)
    count = 1
    while fullname.exists():
        fullname = Path(fulldir, f"{savename}_{count}").with_suffix(ext)
        count += 1 

    return fullname


#############################################
def get_df_label_vals(df, label, vals=None):
    """
    get_df_label_vals(df, label)

    Returns values for a specific label in a dataframe. If the vals is "any", 
    "all" or None, returns all different values for that label.
    Otherwise, vals are returned as a list.

    Required args:
        - df (pandas df): dataframe
        - label (str)   : label of the dataframe column of interest

    Optional args:
        - val (str or list): values to return. If val is None, "any" or "all", 
                             all values are returned.
                             default=None
    Return:
        - vals (list): values
    """

    if vals in [None, "any", "all"]:
        vals = df[label].unique().tolist()
    elif not isinstance(vals, list):
        vals = [vals]
    return vals
    
    
#############################################
def get_df_unique_vals(df, axis="index"):
    """
    get_df_unique_vals(df)

    Returns a list of unique values for each level of the requested axis, in 
    hierarchical order.

    Required args:
        - df (pd.DataFrame): hierarchical dataframe
    
    Optional args:
        - axis (str): Axis for which to return unique values ("index" or 
                      "columns")
                      default: "index"

    Returns:
        - unique_vals (list): unique values for each index or column level, in 
                              hierarchical order
    """

    if axis in ["ind", "idx", "index"]:
        unique_vals = [df.index.unique(row) for row in df.index.names]
    elif axis in ["col", "cols", "columns"]:
        unique_vals = [df.columns.unique(col) for col in df.columns.names]
    else:
        accepted_values_error("axis", axis, ["index", "columns"])


    return unique_vals


#############################################
def reshape_df_data(df, squeeze_rows=False, squeeze_cols=False):
    """
    reshape_df_data(df)

    Returns data array extracted from dataframe and reshaped into as many
    axes as index/column levels, if possible, in hierarchical order.

    Required args:
        - df (pd.DataFrame): hierarchical dataframe
    
    Optional args:
        - squeeze_rows (bool): if True, rows of length 1 are squeezed out
                               default: False
        - squeeze_cols (bool): if True, columns of length 1 are squeezed out
]                              default: False

    Returns:
        - df_data (nd array): dataframe data reshaped into an array
    """

    row_dims = [len(df.index.unique(row)) for row in df.index.names]
    col_dims = [len(df.columns.unique(col)) for col in df.columns.names]

    if squeeze_rows:
        row_dims = filter(lambda dim: dim != 1, row_dims)
    if squeeze_cols:
        col_dims = filter(lambda dim: dim != 1, col_dims)

    new_dims = [*row_dims, *col_dims]

    if np.prod(new_dims) != df.size:
        raise RuntimeError("Unable to automatically reshape dataframe data, as "
            "levels are not shared across all labels.")

    df_data = df.to_numpy().reshape(new_dims)

    return df_data


#############################################
def set_object_columns(df, cols, in_place=False):
    """
    set_object_columns(df, cols)

    Returns dataframe with columns converted to object columns. If a column 
    does not exist, it is created first.

    Required args:
        - df (pandas df): dataframe
        - cols (list)   : list of columns to convert to or create as 
                          object columns 

    Optional args:
        - in_place (bool): if True, changes are made in place. Otherwise, a 
                           deep copy of the dataframe is made first.
                           default: False
    """

    if not in_place:
        df = df.copy(deep=True)

    if not isinstance(cols, list):
        cols = [cols]

    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].astype(object)

    return df
    
    
#############################################
def get_closest_idx(targ_vals, src_vals, allow_out=False):
    """
    get_closest_idx(targ_vals, src_vals)

    Returns index of closest value in targ_vals for each value in src_vals. 

    Required args:
        - targ_vals (1D array): target array of values
        - src_vals (1D array): values for which to find closest value in 
                               targ_vals

    Returns:
        - idxs (1D): array with same length as src_vals, identifying closest 
                     values in targ_vals
    """

    targ_vals = np.asarray(targ_vals)
    src_vals = np.asarray(src_vals)

    if (np.argsort(targ_vals) != np.arange(len(targ_vals))).any():
        raise RuntimeError("Expected all targ_vals to be sorted.")

    if (np.argsort(src_vals) != np.arange(len(src_vals))).any():
        raise RuntimeError("Expected all src_vals to be sorted.")

    idxs = np.searchsorted(targ_vals, src_vals)
    idxs_incr = idxs - 1

    min_val = np.where(idxs_incr == -1)[0]
    if len(min_val):
        idxs_incr[min_val] = 0

    max_val = np.where(idxs == len(targ_vals))[0]
    if len(max_val):
        idxs[max_val] = len(targ_vals) - 1

    all_vals = np.stack([targ_vals[idxs], targ_vals[idxs_incr]])
    val_diff = np.absolute(all_vals - src_vals.reshape(1, -1))

    all_idxs = np.stack([idxs, idxs_incr]).T
    idxs = all_idxs[np.arange(len(all_idxs)), np.argmin(val_diff, axis=0)]

    return idxs
    
    
#############################################
def slice_idx(axis, pos):
    """
    slice_idx(axis, pos)

    Returns a tuple to index an array based on an axis and position on that
    axis.

    Required args:
        - axis (int)            : axis number (non negative)
        - pos (int, list, slice): position(s) on axis

    Returns:
        - sl_idx (slice): slice corresponding to axis and position passed.
    """

    if axis is None and pos is None:
        sl_idx = tuple([slice(None)])

    elif axis < 0:
        raise NotImplementedError("Negative axis values not accepted, as "
            "they are not correctly differentiated from 0.")

    else:
        sl_idx = tuple([slice(None)] * axis + [pos])

    return sl_idx
    
    
#############################################
def remove_idx(items, rem, axis=0):
    """
    remove_idx(items, rem)

    Returns input with items at specific indices in a specified axis removed.

    Required args:
        - items (item or array-like): array or list from which to remove 
                                      elements
        - rem (item or array-like)  : list of idx to remove from items

    Optional args:
        - axis (int): axis along which to remove indices if items is an array
                      default: 0

    Returns:
        - items (array-like): list or array with specified items removed.
    """

    if not isinstance(rem, list):
        rem = [rem]

    if isinstance(items, list):
        make_list = True
        items     = np.asarray(items, dtype=object)
    else:
        make_list = False

    all_idx = items.shape[axis]
    keep = sorted(set(range(all_idx)) - set(rem))
    keep_slice = slice_idx(axis, keep)

    items = items[keep_slice]

    if make_list:
        items = items.tolist()
    
    return items
    
    
#############################################
def num_ranges(ns, pre=0, leng=10):
    """
    num_ranges(ns)

    Returns all indices within the specified range of the provided reference 
    indices. 

    Required args:
        - ns (list): list of reference numbers

    Optional args:
        - pre (num) : indices to include before reference to include
                      default: 0
        - leng (num): length of range
                      default: 10
    Returns:
        - num_ran (2D array): array of indices where each row is the range
                              around one of the input numbers (ns x ranges)
    """

    post = float(leng) - pre

    pre, post = [int(np.around(p)) for p in [pre, post]]

    num_ran = np.asarray([list(range(n-pre, n+post)) for n in ns])

    return num_ran


#############################################
def seed_all(seed=None, seed_now=True):
    """
    seed_all()

    Seeds random number generator using the seed provided or a randomly 
    generated seed if no seed is given.

    Optional args:
        - seed (int or None): seed value to use. (-1 treated as None)
                              default: None
        - seed_now (bool)   : if True, random number generators are seeded now
                              default: True
    Returns:
        - seed (int): seed value
    """
 
    if seed in [None, -1]:
        MAX_INT32 = 2**32
        seed = np.random.randint(1, MAX_INT32)
    
    if seed_now:
        np.random.seed(seed)

    
    return seed


#############################################
def keep_dict_keys(in_dict, keep_if):
    """
    keep_dict_keys(in_dict, keep_if)

    Returns dictionary with only specified keys retained, if they are present.

    Required args:
        - in_dict (dict): input dictionary
        - keep_if (list): list of keys to keep if they are in the input 
                          dictionary
    
    Returns:
        - out_dict (dict): dictionary with keys retained
    """

    out_dict = dict()
    for key in keep_if:
        if key in in_dict.keys():
            out_dict[key] = in_dict[key]

    return out_dict


#############################################
def get_n_cores(parallel=True, max_cores="all"):
    """
    get_n_cores()

    Returns number of cores available for parallel processes.
    
    Optional args:
        - parallel (bool)       : if False, n_jobs of None is returned
                                  default: True
        - max_cores (str or num): max number or proportion of cores to use 
                                  ("all", proportion or int)
                                  default: "all"

    Returns:
        - n_cores (int): number of cores that are usable (None if not 
                         parallel)
    """

    if not parallel:
        n_cores = None

    else:
        n_cores = multiprocessing.cpu_count()
        if max_cores != "all":
            max_cores = float(max_cores)
            if max_cores >= 0.0 and max_cores <= 1.0:
                n_cores = int(n_cores * max_cores)
            else:
                n_cores = min(n_cores, max_cores)

        # check for an environment variable setting the number of threads to
        # use for parallel processes
        max_cores_env = os.getenv("OMP_NUM_THREADS")
        if os.getenv("OMP_NUM_THREADS") is not None:
            n_cores = min(n_cores, int(max_cores_env))

        n_cores = int(max(1, n_cores)) # at least 1

    return n_cores
    
    
#############################################
def get_n_jobs(n_tasks, parallel=True, max_cores="all"):
    """
    get_n_jobs(n_tasks)

    Returns number of jobs corresponding to the criteria passed.

    Required args:
        - n_tasks (int): number of tasks to run
    
    Optional args:
        - parallel (bool)       : if False, n_jobs of None is returned
                                  default: True
        - max_cores (str or num): max number or proportion of cores to use 
                                  ("all", proportion or int)
                                  default: "all"

    Returns:
        - n_jobs (int): number of jobs to use (None if not parallel or fewer 
                        than 2 jobs calculated)
    """

    if not parallel:
        n_jobs = None

    else:
        n_cores = get_n_cores(parallel, max_cores)
        n_jobs = min(int(n_tasks), n_cores)
        if n_jobs < 2:
            n_jobs = None

    return n_jobs


#############################################
class ProgressParallel(Parallel):
    """
    Class allowing joblib Parallel to work with tqdm.
    
    Taken from https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib.
    """

    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        """
        Initializes a joblib Parallel object that works with tqdm.

        Optional args:
            - use_tqdm (bool): if True, tqdm is used
                               default: True
            - total (int)    : number of items in the progress bar
                               default: None
        """

        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        from tqdm.auto import tqdm
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
        
        
#############################################
class ParallelLogging(logger_util.StoreRootLoggingInfo):
    """
    Context manager for temporarily storing root logging information in global 
    variables. Allows root logger handlers and level to be preserved within 
    parallel processes, e.g., if using the loky backend.

    see StoreRootLoggingInfo
    """

    def __init__(self, **kwargs):

        extra_warn_msg = " May not be preserved in parallel processes."

        super().__init__(extra_warn_msg=extra_warn_msg, **kwargs)
        
        
#############################################
def parallel_wrap(fct, loop_arg, args_list=None, args_dict=None, parallel=True, 
                  max_cores="all", zip_output=False, mult_loop=False, 
                  pass_parallel=False, use_tqdm=False):
    """
    parallel_wrap(fct, loop_arg)

    Wraps functions to run them in parallel if parallel is True (not 
    implemented as a python wrapper, to enable additional flexibility).

    Required args:
        - fct (function) : python function
        - loop_arg (list): argument(s) over which to loop (must be first 
                           arguments of fct)
                           if multiple arguments, they must already be zipped 
                           (where the length is the number of items to loop 
                           over), and mult_loop must be set to True
    
    Optional args:
        - args_list (list)      : function input argument list    
                                  default: None
        - args_dict (dict)      : function input argument dictionary
                                  default: None
        - parallel (bool)       : if False, n_jobs of None is returned
                                  default: True
        - max_cores (str or num): max number or proportion of cores to use 
                                  ("all", proportion or int)
                                  default: "all"
        - zip_output (bool)     : if True, outputs are zipped, and tuples are
                                  converted to lists
                                  default: False
        - mult_loop (bool)      : if True, the loop argument contains multiple 
                                  consecutive first arguments
        - pass_parallel (bool)  : if True, 'parallel' argument is passed to the 
                                  function to ensure that 
                                  (1) if this function does run in parallel, 
                                  subfunctions will not sprout parallel joblib 
                                  processes.
                                  (2) is this function does not run in 
                                  parallel, the value of 'parallel' is still 
                                  passed on.
                                  default: False
        - use_tqdm (bool)       : if True, tqdm is used for progress bars.
                                  default: False

    Returns:
        - outputs (list of tuples): outputs, structured as 
                                        (loop_arg length) x 
                                        (number of output values), 
                                    or if zip_output, structured as 
                                        (number of output values) x 
                                        (loop_arg length)
    """

    loop_arg = list(loop_arg)
    n_jobs = get_n_jobs(len(loop_arg), parallel, max_cores)
    
    if args_list is None: 
        args_list = []
    elif not isinstance(args_list, list):
        args_list = [args_list]

    # to allow multiple arguments to be looped over (mimicks zipping)
    if not mult_loop:
        loop_arg = [(arg, ) for arg in loop_arg]

    # enable information to be passed to the function as to whether it can 
    # sprout parallel processes
    if pass_parallel and args_dict is None:
        args_dict = dict()

    if n_jobs is not None and n_jobs > 1:
        from matplotlib import pyplot as plt
        plt.close("all") # prevent garbage collection problems
        
        if use_tqdm:
            ParallelUse = ProgressParallel(
                use_tqdm=True, total=len(loop_arg), n_jobs=n_jobs
                )
        else:
            # multiprocessing backend to enable proper logging
            ParallelUse = Parallel(n_jobs=n_jobs)

        if pass_parallel: 
            # prevent subfunctions from also sprouting parallel processes
            args_dict["parallel"] = False 
        if args_dict is None:
            with ParallelLogging():
                outputs = ParallelUse(
                    delayed(fct)(*arg, *args_list) for arg in loop_arg
                    )
        else:
            with ParallelLogging():
                outputs = ParallelUse(
                    delayed(fct)(*arg, *args_list, **args_dict) 
                    for arg in loop_arg
                    )
    else:
        if pass_parallel: # pass parallel on
            args_dict["parallel"] = parallel
        if use_tqdm:
            from tqdm import tqdm
            loop_arg = tqdm(loop_arg)

        outputs = []
        if args_dict is None:
            for arg in loop_arg:
                outputs.append(fct(*arg, *args_list))
        else:
            for arg in loop_arg:
                outputs.append(fct(*arg, *args_list, **args_dict))

    if zip_output:
        outputs = [*zip(*outputs)]

    return outputs

