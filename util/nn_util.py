"""
nn_util.py

Classes to support the training of neural network models on data from
the AIBS OpenScope Credit Assignment project.

Authors: Blake Richards

Date: September, 2018

Note: this code uses python 3.7.

"""
import argparse as ap
import os
import pdb
import random as rnd

import numpy as np
import tables as tb
import torch
import torchvision


#########################################################################
class SessionROIDataset(torch.utils.data.Dataset):
	"""
	A class for creating pytorch datasets of roi traces for a 
        single session.
	"""

	#################################	
	def __init__(self, session, frames, labels=None, scale=True):
        """
        __init__(session, frames, labels=None, scale=True)

        Create the new dataset object from the session and frames provided. If
        labels is provided, also returns labels with frames (for supervised
        learning).

        Required arguments:
            - session  (Session): 
            - sessionid (string): the ID for this session.

        Optional arguments:
            - droptol (num)  : the tolerance for percentage stimulus frames 
                               dropped, create a Warning if this condition 
                               isn't met.
                               default: 0.0003 
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
#	def __init__(self, session, frames, labels=None, scale=True):
#        """
#        __init__(session, frames, labels=None, scale=True)
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
#            - droptol (num)  : the tolerance for percentage stimulus frames 
#                               dropped, create a Warning if this condition 
#                               isn't met.
#                               default: 0.0003 
#        """
#
#	#################################	
#	def __len__(self):
#
#	#################################	
#	def __getitem__(self, idx):
