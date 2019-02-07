"""
nn_util.py

Classes to support the training of neural network models on data from
the AIBS OpenScope Credit Assignment project.

Authors: Blake Richards

Date: September, 2018

Note: this code uses python 2.7.

"""
import os
import pdb
import argparse as ap
import random as rnd

import tables as tb
import torch
import torchvision
import numpy as np

import session
import gen_util, math_util


#########################################################################
class SessionROIDataset(torch.utils.data.Dataset):
	"""
	A class for creating pytorch datasets of roi traces for a 
        single session.
	"""

	#################################	
	def __init__(self, session, frames, labels=None, normalize=True):
        """
        __init__(session, frames, labels=None, normalize=True)

        Create the new dataset object from the session and frames provided. If
        labels is provided, also returns labels with frames (for supervised
        learning).

        Required arguments:
            - session  (Session): 
            - sessionid (string): the ID for this session.

        Optional arguments:
            - droptol (float): the tolerance for percentage stimulus frames 
                               dropped, create a Warning if this condition 
                               isn't met.
                               default = 0.0003 
        """

	#################################	
	def __len__(self):

	#################################	
	def __getitem__(self, idx):

#########################################################################
#class SessionRawDataset(torch.utils.data.Dataset):
#	"""
#	A class for creating pytorch datasets of raw 2p images for a 
#        single session.
#	"""
#
#	#################################	
#	def __init__(self, session, frames, labels=None, normalize=True):
#        """
#        __init__(session, frames, labels=None, normalize=True)
#
#        Create the new dataset object from the session and frames provided. If
#        labels is provided, also returns labels with frames (for supervised
#        learning).
#
#        Required arguments:
#            - session  (Session): 
#            - sessionid (string): the ID for this session.
#
#        Optional arguments:
#            - droptol (float): the tolerance for percentage stimulus frames 
#                               dropped, create a Warning if this condition 
#                               isn't met.
#                               default = 0.0003 
#        """
#
#	#################################	
#	def __len__(self):
#
#	#################################	
#	def __getitem__(self, idx):
