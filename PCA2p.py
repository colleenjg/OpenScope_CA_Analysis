'''
PCA2p.py

This module contains functions for performing principal components analysis on
2-photon data generated by the AIBS experiments for the Credit Assignment Project.

Authors: Joel Zylberberg, Blake Richards, Colleen Gillon

Date: August, 2018

Note: this code uses python 2.7.

'''

import h5py
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import scipy.stats as stats
from sklearn.decomposition import PCA, RandomizedPCA
import exceptions
import pdb

###############################################################################
def run_pca(datafile, outfile, ncomps = 20, nframes=500, range=(1000,-1000), crop=(500, 500)):
    '''
    run_pca(data_file, pca_file, frames=(1000,-1000), crop=(500,500))

    Runs principal components analysis on the raw 2p imaging data. Stores the results
    in a pickle file in the data directory.

    Required arguments:
            - data_file  (string): full path name of the motion corrected data
            - pca_file (string)  : full path name for the pickle file to put the PCA results in

    Optional arguments:
            - ncomps (int)           : the number of principle components to extract
                                       default = 20
            - nframes (int)          : the number of frames to include for calculating the
                                       principal components, if less than the max range,
                                       will use linear spacing
                                       default = 500
            - range (2-tuple of ints): the range of frames to include in the analysis,
                                       if the second number is negative, the analysis 
                                       will include frames up to that number away
                                       from the total number of frames
                                       default = (1000,-1000)
            - crop (2-tuple of ints) : the size of the image frame to take, e.g. the
                                       crop size (note: max is (512,512)
                                       default = (500,500)

    Outputs:
        - principle_components (array): array of the principle components, with shape
                                        [ncomps, crop[0]*crop[1]]
    '''

    # open the data file
    try:
        fdata = h5py.File(datafile,'r')
    except:
        raise exceptions.IOError('Could not open %s for reading' %datafile)

    # determine the total number of frames in the recording
    totframes = len(fdata['data']);

    # get a list of all the frames we'll use here
    if range[1] <= 0:
        flist = [int(i) for i in np.ceil(np.linspace(range[0], totframes+range[1], nframes)).tolist()]
    elif range[1] <= range[0]:
        raise exceptions.UserWarning('Second element of range must either be negative or greater than first element, reverting to default')
        range = [1000,-1000]
        flist = [int(i) for i in np.ceil(np.linspace(range[0], totframes+range[1], nframes)).tolist()]
    else:
        flist = [int(i) for i in np.ceil(np.linspace(range[0], range[1], nframes)).tolist()]

    # get the requested frames (cropped) into a numpy array reshaped for PCA
    data = np.array(fdata['data'][flist,0:crop[0],0:crop[1]]).reshape([nframes, crop[0]*crop[1]])

    # get the fluoescence baseline for each pixel as the mean
    baselineF = np.mean(data,axis=0)

    # run the principle components analysis
    princomps = PCA(ncomps,svd_solver='randomized').fit(data)

    # create a dictionary
    pcadict = {'baseline': baselineF, 'pca': princomps, 'crop': crop}

    # save the pickle file
    with open(outfile,"wb") as output_file:
        pickle.dump(pcadict,output_file)

    return princomps.components_

###############################################################################
def project_data(datafile, pcafile, frames):
    '''
    project_data(data_file, pca_file, frames)

    Projects the 2p data in frames onto the principle components stored in pca_file.

    Required arguments:
            - data_file  (string)  : full path name of the motion corrected data
            - pca_file (string)    : full path name for the pickle file with the PCA results
            - frames (list of ints): frames to extract data from and project with

    Outputs:
        - projection (array): array of the data projected onto the principle components
                              shape is [len(frames), crop[0]*crop[1]] where crop is pulled
                              from pca_file
    '''

    # open the data file
    try:
        fdata = h5py.File(datafile,'r')
    except:
        raise exceptions.IOError('Could not open %s for reading' %datafile)

    # open the pickle file
    try:
        pklfile = open(pcafile,'rb')
        pcapkl  = pickle.load(pklfile)
        pklfile.close()
    except:
        raise exceptions.IOError('Could not open %s for reading' %pcafile)

    # get the crop info
    crop = pcapkl['crop']

    # get the baseline fluorescence
    #baseline = pcapkl['baseline']
    #pdb.set_trace()

    # get the pca components
    comps = pcapkl['pca'].components_

    # initialize the return array
    proj = np.zeros((comps.shape[0],len(frames)))

    # for each frame, project the data
    for f in frames:
        proj[:,f] = np.dot(comps,np.array(fdata['data'][f,0:crop[0],0:crop[1]]).reshape([crop[0]*crop[1],1])).squeeze()

    return proj
        