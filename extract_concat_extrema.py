import argparse
import glob
import multiprocessing

import h5py
import numpy as np
from joblib import Parallel, delayed


def extract_extrema(concat, block_size=100):

    frame_min = np.inf
    frame_max = -np.inf
    try:
        with h5py.File(concat, 'r') as f:
            num_frames = f['data'].shape[0]
            for frame_num in range(0, num_frames, block_size):
                new_min = np.min(f['data'][frame_num:frame_num+block_size])
                new_max = np.max(f['data'][frame_num:frame_num+block_size])

                if new_min < frame_min:
                    frame_min = new_min
                if new_max > frame_max:
                    frame_max = new_max
        print('\n Min: {}, Max: {} for {}'.format(frame_min, frame_max, concat))

    except Exception as err:
        print('\nFailed for {} due to {}'.format(concat, err))

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
        # general parameters
    parser.add_argument('--datadir', default=None, 
                        help=('data directory'))
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--block_size', default=100, type=int)
    args = parser.parse_args()


    all_concat = glob.glob('{}/*/*/*/*/*/concat_31Hz_0.h5'.format(args.datadir))

    print('{} concat files'.format(len(all_concat)))

    if args.parallel:
        n_cores = multiprocessing.cpu_count()
        n_jobs = min(n_cores, len(all_concat))

        Parallel(n_jobs=n_jobs)(delayed(extract_extrema)(concat, args.block_size) 
                for concat in all_concat)

    else:
        for concat in all_concat:
            extract_extrema(concat, args.block_size)
           

