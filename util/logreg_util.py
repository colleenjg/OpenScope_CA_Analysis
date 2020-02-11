"""
logreg_util.py

Functions and classes for logistic regressions.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import copy
import glob
import os
import re

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import torch

from util import file_util, gen_util, math_util, plot_util


#############################################
class LogReg(torch.nn.Module):
    """
    The LogReg object is a pytorch Neural Network module object that 
    implements a logistic regression.
    """
    
    def __init__(self, num_units, num_fr):
        """
        self.__init__(num_units, num_fr)

        Initializes and returns the new LogReg object using the specified 2D 
        input dimensions which are flattened to form a 1D input layer. 
        
        The network is composed of a single linear layer from the input layer 
        to a single output on which a sigmoid is applied.

        Initializes num_units, num_fr, lin and sig attributes.

        Required args:
            - num_units (int): nbr of units.
            - num_fr (int)   : nbr of frames per unit.
        """

        super(LogReg, self).__init__()
        self.num_units = num_units
        self.num_fr    = num_fr
        self.lin = torch.nn.Linear(self.num_units*self.num_fr, 1)
        self.sig = torch.nn.Sigmoid()
        
    def forward(self, x):
        x_resh = x.view(-1, self.num_units*self.num_fr)
        return self.sig(self.lin(x_resh)).view(-1, 1)


#############################################        
class weighted_BCE():
    """
    The weighted_BCE object defines a weighted binary crossentropy (BCE) loss.
    """

    def __init__(self, weights=None):
        """
        self.__init__()

        Initializes and returns the new weighted_BCE object using the specified 
        weights. 
        
        Initializes weights, name attributes.

        Optional args:
            - weights (list): list of weights for both classes [class0, class1] 
                              default: None
        """

        if weights is not None and len(weights) != 2:
            raise ValueError('Exactly 2 weights must be provided, if any.')
        
        self.weights = weights
        self.name = 'Weighted BCE loss'

    def calc(self, pred_class, act_class):
        """
        self.calc(pred_class, act_class)

        Returns the weighted BCE loss between the predicted and actual 
        classes using the weights.
        
        Required args:
            - pred_class (nd torch Tensor): array of predicted classes 
                                            (0 and 1s)
            - act_class (nd torch Tensor) : array of actual classes (0 and 1s)

        Returns:
            - BCE (torch Tensor): single BCE value
        """

        if self.weights is not None:
            weights = act_class*(self.weights[1]-self.weights[0]) + \
                      (self.weights[0])
        else:
            weights = None

        BCE = torch.nn.functional.binary_cross_entropy(pred_class, act_class, 
                                                       weight=weights)
        return BCE



#############################################
#############################################
def class_weights(train_classes):
    """
    class_weights(train_classes)

    Returns the weights for classes, based on their proportions in the
    training class as: train_len/(n_classes * n_class_values).
    
    Required args:
        - train_class (nd array): array of training classes

    Returns:
        - weights (list): list of weights for each class

    """

    train_classes = np.asarray(train_classes).squeeze()
    classes = list(np.unique(train_classes))
    weights = []
    for cl in classes:
        weights.append((len(train_classes)/(float(len(classes)) *
                        list(train_classes).count(cl))))
    
    return weights


#############################################
def accuracy(pred_class, act_class):
    """
    accuracy(pred_class, act_class)

    Returns the accuracy for each class (max 2), and returns the actual number 
    of samples for each class as well as the accuracy.
    
    Required args:
        - pred_class (nd array): array of predicted classes (0 and 1s)
        - act_class (nd array) : array of actual classes (0 and 1s)

    Returns:
        - (list): 
            - n_class0 (int): number of class 0 samples
            - n_class1 (int): number of class 1 samples
        - (list):
            - acc_class0 (num)  : number of class 0 samples correctly predicted
            - acc_class1 (num)  : number of class 1 samples correctly predicted
    """

    act_class  = np.asarray(act_class).squeeze()
    pred_class = np.round(np.asarray(pred_class)).squeeze()
    n_class1 = sum(act_class)
    n_class0 = len(act_class) - n_class1
    if n_class1 != 0:
        acc_class1  = list(act_class + pred_class).count(2)
    else:
        acc_class1 = 0
    if n_class0 != 0:
        acc_class0  = list(act_class + pred_class).count(0)
    else:
        acc_class0 = 0
    return [n_class0, n_class1], [acc_class0, acc_class1]


#############################################
def get_sc_types(info='label'):
    """
    get_sc_types()

    Returns info about the four score types: either labels, titles, list 
    indices, or lists to track scores within an epoch.
    
    Optional args:
        - info (str)  : type of info to return (label, title, idx or track)
                        default: 'label'

    Returns:
        if info == 'label':
            - label (list): list of score type labels
        elif info == 'title':
            - title (list): list of score type titles
    """

    label = ['loss', 'acc', 'acc_class0', 'acc_class1', 'acc_bal']
    
    if info == 'label':
        return label

    elif info == 'title':
        title = ['Loss', 'Accuracy (%)', 'Accuracy on class0 trials (%)', 
                 'Accuracy on class1 trials (%)', 'Balanced accuracy (%)']
        return title


#############################################
def get_sc_names(loss_name, classes):
    """
    get_sc_names(loss_name, classes)

    Returns specific score names, incorporating name of type of loss function
    and the class names.
    
    Required args:
        - loss_name (str): name of loss function
        - classes (list) : list of names of each class

    Returns:
        - sc_names (list): list of specific score names
    """

    sc_names = get_sc_types(info='title')
    for i, sc_name in enumerate(sc_names):
        if sc_name.lower() == 'loss':
            sc_names[i] = loss_name
        for j, class_name in enumerate(classes):
            generic = 'class{}'.format(j)
            if generic in sc_name:
                sc_names[i] = sc_name.replace(generic, class_name)
    
    return sc_names


#############################################
def get_set_labs(test=True, ext_test=False, ext_test_name=None):
    """
    get_set_labs()

    Returns labels for each set (train, val, test).
    
    Optional args:
        - test (bool )       : if True, a test set is included
                               default: True
        - ext_test (bool)    : if True, an extra test set is included
                               default: False
        - ext_test_name (str): name of extra test set, if included
                               default: None

    Returns:
        - sets (list): list of set labels
    """

    sets = ['train', 'val']
    if test:
        sets.extend(['test'])
    if ext_test:
        if ext_test_name is None:
            sets.extend(['ext_test'])
        else:
            sets.extend([ext_test_name])
    
    return sets


#############################################
def get_sc_labs(test=True, by='flat', ext_test=False, ext_test_name=None):
    """
    get_sc_labs()

    Returns labels for each set (train, val, test).
    
    Optional args:
        - test (bool)        : if True, a test set is included
                               default: True
        - by (str)           : if 'flat', labels are returned flat. If 'set', 
                               labels are returned by set.
                               default: 'flat'
        - ext_test (bool)    : if True, an extra test set is included
                               default: False
        - ext_test_name (str): name of extra test set, if included
                               default: None

    Returns:
        - sc_labs (list): list of set and score labels, 
                          nested by set if by is 'set'.
    """

    if ext_test_name is not None:
        ext_test = True

    sets = get_set_labs(test, ext_test=ext_test, ext_test_name=ext_test_name)
    scores = get_sc_types()
    if by == 'flat':
        sc_labs = ['{}_{}'.format(s, sc) for s in sets for sc in scores]
    elif by == 'set':
        sc_labs = [['{}_{}'.format(s, sc) for sc in scores] for s in sets]
    else:
        gen_util.accepted_values_error('by', by, ['flat', 'set'])

    return sc_labs


#############################################
def run_batches(mod, dl, device, train=True):
    """
    run_batches(mod, dl, device)

    Runs dataloader batches through network and returns scores.
    
    Required args:
        - mod (torch.nn.Module): Neural network module with optimizer and loss 
                                 function as attributes
        - dl (torch DataLoader): Dataloader
        - device (str)         : device to use ('cuda' or 'cpu') 

    Optional args:
        - train (bool): if True, network is trained on data. If False, 
                        network is evaluated on data, but not trained.
                        default: True

    Returns:
        - ep_sc (dict): dictionary of epoch scores (loss, acc, acc_class0, 
                        acc_class1)
    """

    labs = get_sc_types('label')
    ep_sc, divs = dict(), dict()

    for lab in labs:
        ep_sc[lab] = 0
        if lab in ['loss', 'acc']:
            divs[lab] = dl.dataset.n_samples
        else:
            divs[lab] = 0

    for _, (data, targ) in enumerate(dl, 0):
        if train:
            mod.opt.zero_grad()
        pred_class = mod(data.to(device))
        loss = mod.loss_fn.calc(pred_class, targ.to(device))
        if train:
            loss.backward()
            mod.opt.step()
        # retrieve sum across batch
        ep_sc['loss'] += loss.item()*len(data) 
        ns, accs = accuracy(pred_class.cpu().detach(), targ.cpu().detach())
        ep_sc['acc'] += accs[0] + accs[1]
        for lab, n, acc in zip(labs[-3:-1], ns, accs):
            if acc is not None:
                ep_sc[lab] += acc
                divs[lab]  += float(n)
    
    for lab in labs:
        mult = 1.0
        if 'acc' in lab:
            mult = 100.0
        if lab != 'acc_bal':
            ep_sc[lab] = ep_sc[lab] * mult/divs[lab]
    
    cl_accs = []
    if 'acc_bal' in labs: # get balanced accuracy (across classes)
        for lab in labs:
            if 'class' in lab:
                cl_accs.append(ep_sc[lab])
        if len(cl_accs) > 0:
            ep_sc['acc_bal'] = np.mean(cl_accs) 
        else:
            raise ValueError(('No class accuracies. Cannot calculate '
                              'balanced accuracy.'))
    
    return ep_sc


#############################################
def run_dl(mod, dl, device, train=True):
    """
    run_dl(mod, dl, device)

    Sets model to train or evaluate, runs dataloader through network and 
    returns scores.
    
    Required args:
        - mod (torch.nn.Module): Neural network module with optimizer and loss 
                                 function as attributes
        - dl (torch DataLoader): Dataloader
        - device (str)         : device to use ('cuda' or 'cpu') 

    Optional args:
        - train (bool): if True, network is trained on data. If False, 
                        network is evaluated on data, but not trained.
                        default: True

    Returns:
        - ep_sc (dict): dictionary of epoch scores (loss, acc, acc_class0, 
                        acc_class1)
    """

    if train:
        mod.train()
    else:
        mod.eval()
    
    if train:
        ep_sc = run_batches(mod, dl, device, train)
    else:
        with torch.no_grad():
            ep_sc = run_batches(mod, dl, device, train)
    
    return ep_sc


#############################################
def print_loss(s, loss, logger=None):
    """
    print_loss(s, loss)

    Prints or logs at info level loss for set to console.
    
    Required args:
        - s (str)     : set (e.g., 'train')
        - loss (num)  : loss score
    
    Optional args:
        - logger (logger): logger to use. If logger is None, loss is printed to 
                           console.
    """

    print_str = '    {} loss: {:.4f}'.format(s, loss)
    if logger is None:
        print(print_str)
    else:
        logger.info(print_str)


#############################################
def save_model(info, ep, mod, scores, dirname='.', rectype=None): 
    """
    save_model(info, ep, mod, scores)

    Saves model and optimizer, as well as a dictionary with info and epoch 
    scores.
    
    Required args:
        - info (dict)          : dictionary of info to save along with model
        - ep (int)             : epoch number
        - mod (torch.nn.Module): Neural network module with optimizer as 
                                 attribute
        - scores (dict)        : epoch score dictionary, where keys are a
                                 combination of: train, val, test x 
                                    loss, acc, acc_class0, acc_class1

    Optional args:
        - dirname (str): directory in which to save
                         default: '.'
        - rectype (str): type of model being recorded, i.e., 'best' or 'max'
                         If 'best', the previous best models are removed and
                         'best' is included in the name of the recorded model.
                         default: None
    """

    if rectype == 'best':
        # delete previous model
        prev_model = glob.glob(os.path.join(dirname, 'ep*_best.pth'))
        prev_json = glob.glob(os.path.join(dirname, 'ep*_best.json'))
        
        if len(prev_model) == 1 and len(prev_json) == 1:
            os.remove(prev_model[0])
            os.remove(prev_json[0])
        savename = 'ep{}_best'.format(ep)

    else:
        savename = 'ep{}'.format(ep)

    savefile = os.path.join('{}'.format(dirname), savename)
    
    torch.save({'net': mod.state_dict(), 'opt': mod.opt.state_dict()},
                '{}.pth'.format(savefile))
    
    info = copy.deepcopy(info)
    info['epoch_n'] = ep
    info['scores'] = scores

    file_util.saveinfo(info, savename, dirname, 'json')
    

#############################################
def fit_model_pt(info, n_epochs, mod, dls, device, dirname='.', ep_freq=50, 
              test_dl2_name=None, logger=None):
    """
    fit_model_pt(info, epochs, mod, dls, device)

    Fits pytorch model to data and evaluates. Logs scores at info level. 
    Returns an array of scores and an array recording which epochs models were 
    saved for.
    
    Required args:
        - info (dict)          : dictionary of info to save along with model
        - n_epochs (int)       : total number of epochs
        - mod (torch.nn.Module): Neural network module with optimizer and loss 
                                 function as attributes
        - dls (list)           : list of Torch Dataloaders
        - device (str)         : device to use ('cuda' or 'cpu') 

    Optional args:
        - dirname (str)      : directory in which to save models and dictionaries
                               default: '.'
        - ep_freq (int)      : frequency at which to print loss to console
                               default: 50
        - test_dl2_name (str): name of extra DataLoader
                               default: None
        - logger (logger)    : logger object
                               default: None

    Returns:
        - scores (pd DataFrame): dataframe in which scores are recorded, with
                                 columns epoch_n, saved_eps and combinations of
                                 sets x score types
    """

    if logger is None:
        logger = gen_util.get_logger('stream', 'loss_logs', level='info')

    test = False
    ext_test = False
    if len(dls) == 4:
        ext_test = True
        test = True
    elif len(dls) == 3:
        if test_dl2_name is not None:
            ext_test = True
        else:
            test = True

    sets = get_set_labs(test, ext_test=ext_test, ext_test_name=test_dl2_name)
    scs = get_sc_types('label')
    col_names = get_sc_labs(test, 'flat', ext_test=ext_test, 
                            ext_test_name=test_dl2_name)
    scores = pd.DataFrame(np.nan, index=list(range(n_epochs)), 
                          columns=col_names)
    scores.insert(0, 'epoch_n', list(range(n_epochs)))
    scores['saved'] = np.zeros([n_epochs], dtype=int)
    
    rectype = None
    min_val = np.inf # value to beat to start recording models
    for ep in range(n_epochs):
        ep_loc = (scores['epoch_n'] == ep)
        ep_sc  = dict()
        for se, dl in zip(sets, dls):
            train = False
            # First train epoch: record untrained model
            if ep !=0 and se == 'train': 
                train = True
            set_sc = run_dl(mod, dl, device, train=train)
            for sc in scs:
                col = '{}_{}'.format(se, sc)
                scores.loc[ep_loc, col] = set_sc[sc]
                ep_sc[col] = set_sc[sc]
        
        # record model if val reaches a new low or if last epoch
        if (scores.loc[ep_loc]['val_loss'].tolist()[0] < min_val):
            rectype = 'best'
            min_val = scores.loc[ep_loc]['val_loss'].tolist()[0]
            scores['saved'] = np.zeros([n_epochs], dtype=int)
        elif ep == n_epochs - 1:
            rectype = 'max'
        if rectype in ['best', 'max']:
            scores.loc[ep_loc, 'saved'] = 1
            save_model(info, ep, mod, ep_sc, dirname, rectype)
            rectype = None
        
        if ep % ep_freq == 0:
            logger.info('Epoch {}'.format(ep))
            print_loss('train', scores.loc[ep_loc, 'train_loss'].tolist()[0], 
                       logger)
            print_loss('val', scores.loc[ep_loc]['val_loss'].tolist()[0], 
                       logger)
    
    return scores


#############################################
def get_epoch_n_pt(dirname, model='best'):
    """
    get_epoch_n_pt(dirname)

    Returns requested recorded epoch number in a directory. Expects models to 
    be recorded as 'ep*.pth', where the digits in the name specify the epoch 
    number.
    
    Required args:
        - dirname (str): directory path

    Optional args:
        - model (str): model to return ('best', 'min' or 'max')
                       default: 'best'

    Returns:
        - ep (int): number of the requested epoch 
    """

    warn_str='===> Warning: '
    ext_str = ''
    if model == 'best':
        ext_str = '_best'
    models = glob.glob(os.path.join(dirname, 'ep*{}.pth'.format(ext_str)))
    
    if len(models) > 0:
        ep_ns = [int(re.findall(r'\d+', os.path.split(mod)[-1])[0]) 
                     for mod in models]
    else:
        print('{} No models were recorded.'.format(warn_str))
        ep = None
        return ep
    
    if model == 'best':
        ep = np.max(ep_ns)
    elif model == 'min':
        ep = np.min(ep_ns)
    elif model == 'max':
        ep = np.max(ep_ns)
    else:
        gen_util.accepted_values_error('model', model, ['best', 'min', 'max'])

    return ep


#############################################
def load_params(dirname, model='best', alg='sklearn'):
    """
    load_params(dirname)

    Returns model parameters: epoch number, model weights and model biases. 
    Expects models to be recorded as 'ep*.pth', where the digits in the name 
    specify the epoch number. 
    
    Required args:
        - dirname (str): directory path

    Optional args:
        - model (str or idx): model to return ('best', 'first' or 'last')
                              default: 'best'
        - alg (str)         : algorithm used to run logistic regression 
                              ('sklearn' or 'pytorch')
                              default: 'sklearn'

    Returns:
        if recorded models are found:
            - ep (int)          : number of the requested epoch 
            - weights (2D array): LogReg network weights, 
                                  structured as 1 x n input values
            - biases  (1D array): LogReg network bias, single value 
        if alg == 'sklearn':
            - model (int)       : model number
        otherwise returns None
    """

    if alg == 'sklearn':
        filename = os.path.join(dirname, 'models.sav')
        if os.path.exists(filename) and not isinstance(model, str):
            with open(filename, 'rb') as f:
                mod = pkl.load(f)['estimator'][model]['logisticregression']
            weights = mod.coef_
            biases  = mod.intercept_
            ep      = mod.n_iter_[0]
            return ep, weights, biases, model
        else:
            return None

    elif alg == 'pytorch':
        ep = get_epoch_n_pt(dirname, model)
        ext_str = ''
        if model == 'best':
            ext_str = '_best'

        if ep is None:
            return None
        else:
            models = glob.glob(os.path.join(dirname, 
                                            'ep{}*{}.pth'.format(ep, ext_str)))[0]
            checkpoint = torch.load(models)
            weights = checkpoint['net']['lin.weight'].numpy()
            biases = checkpoint['net']['lin.bias'].numpy()
            return ep, weights, biases

    else:
        gen_util.accepted_values_error('alg', alg, ['sklearn', 'pytorch'])


#############################################
def load_checkpoint_pt(mod, filename):
    """
    load_checkpoint_pt(mod, filename)

    Returns model updated with recorded parameters and optimizer state. 
    
    Required args:
        - mod (torch.nn.Module): Neural network module with optimizer as 
                                 attribute
        - filename (str)       : name of the file (should be '.pth')

    Returns:
        - mod (torch.nn.Module): Neural network module with model parameters 
                                 and optimizer updated.
    """

    # Note: Input model & optimizer should be pre-defined.  This routine only 
    # updates their states.
    checkpt_name = os.path.split(filename)[-1]
    if os.path.isfile(filename):
        print(('\nLoading checkpoint found at \'{}\''.format(checkpt_name)))
        checkpoint = torch.load(filename)
        mod.load_state_dict(checkpoint['net'])
        mod.opt.load_state_dict(checkpoint['opt'])
    else:
        raise OSError('No checkpoint found at \'{}\''.format(checkpt_name))

    return mod


#############################################
def plot_weights(ax, mod_params, xran, stats='mean', error='sem'):
    """
    plot_weights(mod_params, xran)

    Plots weights on 2 subplots (one by frame, and one by ROI). Also adds
    bias value.
    
    Required args:
        - axs (plt Axis)   : axis (2x2) [across ROIs, across frames]
        - mod_params (list): model parameters [ep_n, weights, bias, 
                             (mod_idx)]
        - xran (1D array)  : array of x range values

    Optional args:
        - stats (str): stats to take, i.e., 'mean' or 'median'
                       default: 'mean'
        - error (str): error to take, i.e., 'std' (for std or 
                       quintiles) or 'sem' (for SEM or MAD)
                       default: 'std'
    """

    if ax.shape != (2, 2):
        raise ValueError(('Axis should be of shape (2, 2), '
                          'but is of shape {}.').format(ax.shape))

    weights = np.asarray(mod_params[1]).reshape(len(xran), -1)

    # plot weights by fr (bottom, left subplot)
    by_fr = ax[1, 0]
    fr_stats = math_util.get_stats(weights, stats, error, axes=1)
    fr_title = 'Model weights (ep {})'.format(mod_params[0])
    if len(mod_params) == 4:
        fr_title = '{} (mod {})'.format(fr_title, mod_params[3])
    plot_util.plot_traces(by_fr, xran, fr_stats[0], fr_stats[1:], fr_title, 
                            col='dimgrey', alpha=0.4)
    by_fr.axhline(y=0, ls='dashed', c='k', lw=1, alpha=0.5)
    orig_tick_max = np.max(np.absolute(by_fr.get_yticks()))
    by_fr.set_yticks([-orig_tick_max, 0, orig_tick_max])

    # plot weights by ROI, sorted (top, right subplot)
    by_roi = ax[0, 1]
    roi_stats = math_util.get_stats(weights, stats, error, axes=0)
    xran_rois = range(roi_stats.shape[1])
    sorter = np.argsort(roi_stats[0])[::-1] # reverse sort
    plot_util.plot_traces(by_roi, roi_stats[0][sorter], xran_rois,
              roi_stats[1:][:, sorter], 'ROI wei', col='dimgrey', 
              alpha=0.4, errx=True)
    by_roi.axvline(x=0, ls='dashed', c='k', lw=1, alpha=0.5)
    orig_tick_max = np.max(np.absolute(by_roi.get_xticks()))
    if orig_tick_max == 0:
        tick_max = np.round(np.max(by_roi.get_xlim()), 2)
        plot_util.set_ticks(by_roi, 'x', -tick_max, tick_max, 3, pad_p=0.1)
    else:
        by_roi.set_xticks([-orig_tick_max, 0, orig_tick_max])

    # write intercept (bottom, right subplot)
    bias_subax = ax[1, 1]
    bias_subax.axis('off')
    bias_text = 'Bias\n{:.2f}'.format(mod_params[2][0])
    bias_subax.text(0.5, 0.5, bias_text, fontsize='x-large', 
               fontweight='bold', ha='center', va='bottom')


#############################################
def get_stats(tr_data, tr_classes, pre=0, post=1.5, classes=None, stats='mean', 
              error='sem'):
    """
    get_stats(tr_data, tr_classes, classes, len_s)

    Plots training data and returns figure, data subplot and trace colors.
    
    Required args:
        - tr_stats (nd array)  : training data array, structured as 
                                 trials x frames x units
        - tr_classes (1D array): training data targets

    Optional args:
        - pre (num)     : start point (number of seconds before 0)
                          default: 0
        - post (num)    : end point (number of seconds after 0)
                          default: 0
        - classes (list): list of class values (if None, inferred from 
                          tr_classes)
                          default: None
        - stats (str)   : stats to take, i.e., 'mean' or 'median'
                          default: 'mean'
        - error (str)   : error to take, i.e., 'std' (for std or quintiles) or 
                          'sem' (for SEM or MAD)
                          default: 'std
    Returns:
        - xran (1D array)     : x values for frames
        - all_stats (3D array): training statistics, structured as 
                                   class x stats (me, err) x frames
        - ns (list)           : number of sequences per class
    """

    xran = np.linspace(-pre, post, tr_data.shape[1])

    ns = []
    # select class trials and take the stats across trials (axis=0), 
    # then across e.g., cells (last axis)
    all_stats = []

    if classes is None:
        classes = np.unique(tr_classes)
    for cl in classes:
        idx = (tr_classes == cl) # bool array
        ns.append(sum(idx.tolist()))
        class_stats = math_util.get_stats(tr_data[idx], stats, error, 
                                          axes=[0, 2])
        all_stats.append(class_stats)
    
    all_stats = np.asarray(all_stats)
    return xran, all_stats, ns


#############################################
def plot_tr_data(xran, class_stats, classes, ns, fig=None, ax_data=None, 
                 plot_wei=True, alg='sklearn', stats='mean', error='sem', 
                 modeldir='.', cols=None, data_type=None, xlabel=None):
    """
    plot_tr_data(xran, class_stats, ns)

    Plots training data, and optionally parameters of the best model. Returns 
    figure, data subplot and trace colors.
    
    Required args:
        - xran (array-like)     : x values for frames
        - class_stats (2D array): statistics for training data array, 
                                  structured as: stat_type (me, err) x frames
        - classes (list)        : list of class names
        - ns (list)             : number of sequences per class

    Optional args:
        - fig (plt fig)         : pyplot figure to plot on. If fig or ax_data 
                                  is None, new ones are created.
                                  default: None
        - ax_data (plt Axis)    : pyplot axis subplot to plot data on. If fig 
                                  or ax_data is None, new ones are created.
                                  default: None
        - plot_wei (bool or int): if True, weights are plotted in a subplot.
                                  Or if int, index of model to plot. Only if 
                                  model to be recorded and no fig or ax_data 
                                  to be passed.
                                  default: True
        - alg (str)             : algorithm used to run logistic regression 
                                  ('sklearn' or 'pytorch')
                                  default: 'sklearn'
        - stats (str)           : stats to take, i.e., 'mean' or 'median'
                                  default: 'mean'
        - error (str)           : error to take, i.e., 'std' (for std or 
                                  quintiles) or 'sem' (for SEM or MAD)
                                  default: 'std
        - dirname (str)         : name of the directory from which to load
                                  model parameters
                                  default: '.'
        - cols (list)           : colors to use
                                  default: None 
        - data_type (str)       : data type if not training (e.g., test)
                                  default: None
        - xlabel (str)          : x axis label
                                  default: None
    
    Returns:
        - fig (plt fig)                : pyplot figure
        - ax_data (pyplot Axis subplot): subplot
        - cols (list)                  : list of trace colors
    """

    model = 'best'
    if not isinstance(plot_wei, bool): # check if it's an int
        model = plot_wei
        plot_wei = True

    if fig is None or ax_data is None:
        # training data: trials x frames x units
        mod_params = load_params(modeldir, model, alg)
        if plot_wei and mod_params is not None:
            fig, ax = plt.subplots(2, 2, figsize=(10, 8), 
                                   gridspec_kw = {'height_ratios':[3, 1], 
                                                  'width_ratios':[4, 1]})
            ax_data = ax[0, 0]
        else:
            fig, ax_data = plt.subplots()
    else:
        plot_wei = False
        mod_params = None

    if cols is None:
        cols = [None] * len(classes)

    if data_type is not None:
        data_str = ' ({})'.format(data_type)
    else:
        data_str = ''

    for i, class_name in enumerate(classes):
        cl_st = np.asarray(class_stats[i])
        leg = '{}{} (n={})'.format(class_name, data_str, ns[i])
        plot_util.plot_traces(ax_data, xran, cl_st[0], cl_st[1:], 
                              alpha=0.8/len(classes), label=leg, col=cols[i])
        cols[i] = ax_data.lines[-1].get_color()

    # plot weights as well
    if plot_wei and mod_params is not None:
        plot_weights(ax, mod_params, xran, stats, error)
        # remove redundant labels
        ax_data.set_xlabel('') 
        ax_data.set_xticklabels([])
    
    if xlabel is not None:
        if plot_wei and mod_params is not None:
            ax[1, 0].set_xlabel(xlabel)
        else:
            ax_data.set_xlabel(xlabel)

    
    return fig, ax_data, cols


#############################################
def plot_scores(scores, classes, alg='sklearn', loss_name='loss', dirname='.', 
                gen_title=''):

    """
    plot_scores(scores, classes)

    Plots each score type in a figure and saves figures.
    
    Required args:
        - scores (pd DataFrame): dataframe in which scores are recorded, for
                                 each epoch (pytorch) or each run (sklearn)
        - classes (list)       : list of class names
    
    Optional args:
        - alg (str)      : algorithm used to run logistic regression 
                           ('sklearn' or 'pytorch')
                           default: 'sklearn'
        - loss_name (str): name of type of loss
                           default: 'loss'
        - dirname (str)  : name of the directory in which to save figure
                           default: '.'
        - gen_title (str): general plot titles
                           default: ''
    """

    if alg == 'pytorch':
        x_vals = list(range(min(scores['epoch_n']), max(scores['epoch_n']) + 1))
        x_lab = 'Epochs'
    elif alg == 'sklearn':
        x_vals = list(range(min(scores['run_n']), max(scores['run_n']) + 1))
        x_lab = 'Runs'
    else:
        gen_util.accepted_values_error('alg', alg, ['pytorch', 'sklearn'])

    sc_labs = get_sc_types('label')
    set_labs, set_names = [], []
    for col_name in scores.keys():
        if sc_labs[0] in col_name:
            set_labs.append(col_name.replace('_{}'.format(sc_labs[0]), ''))
            set_names.append('{} set'.format(set_labs[-1]))

    sc_titles = get_sc_names(loss_name, classes) # for title
    dash_test = 'test_out' in sc_labs

    for sc_title, sc_lab in zip(sc_titles, sc_labs):
        if '{}_{}'.format(set_labs[0], sc_lab) not in scores.keys():
            continue
        fig, ax = plt.subplots(figsize=[20, 5])
        for set_lab in set_labs:
            dashes = (None, None)
            if set_lab == 'train':
                dashes = [3, 2]
            if set_lab == 'val' or (set_lab == 'test' and dash_test):
                dashes = [6, 2]
            sc = np.asarray(scores['{}_{}'.format(set_lab, sc_lab)])
            ax.plot(x_vals, sc, label=set_lab ,lw=2.5, dashes=dashes)
            ax.set_title(u'{}\n{}'.format(gen_title, sc_title))
            ax.set_xlabel(x_lab)
        if 'acc' in sc_lab:
            ax.set_ylim(-5, 105)
        elif 'loss' in sc_lab:
            act_ylim = ax.get_ylim()
            pad = (act_ylim[1] - act_ylim[0]) * 0.05
            ax.set_ylim(act_ylim[0] - pad, act_ylim[1] + pad)
        ax.legend()
        fig.savefig(os.path.join(dirname, '{}'.format(sc_lab)))


#############################################
def check_scores_pt(scores_df, best_ep, hyperpars):
    """
    check_scores_pt(scores_df, best_ep, hyperpars)

    Returns data for the best epoch recorded in scores dataframe for pytorch
    logreg analyses. Also checks that the best epoch in dataframe (based on 
    validation loss) is also the best epoch model saved.
    
    Required args:
        - scores_df (pd DataFrame): scores dataframe
        - best_ep (int)           : max epoch recorded
        - hyperpars (dict)        : dictionary containing hyperparameters

    Returns:
        - ep_info (pd DataFrame): line from score dataframe of max epoch 
                                  recorded.
    """

    warn_str = '===> Warning: '

    ep_info = None

    if scores_df is not None:
        # check that all epochs were recorded and correct epoch
        # was recorded as having lowest validation loss
        ep_rec = scores_df.count(axis=0)
        if min(ep_rec) < hyperpars['logregpar']['n_epochs']:
            print(('{} Only {} epochs were fully '
                    'recorded.').format(warn_str, min(ep_rec)))
        if max(ep_rec) > hyperpars['logregpar']['n_epochs']:
            print(('{} {} epochs were '
                    'recorded.').format(warn_str, max(ep_rec)))
        if len(scores_df.loc[(scores_df['saved'] == 
                              1)]['epoch_n'].tolist()) == 0:
            print(('{} No models were recorded in '
                    'dataframe.').format(warn_str))
        else:
            ep_df = scores_df.loc[(scores_df['saved'] == 1)]['epoch_n'].tolist()
            best_val = np.min(scores_df['val_loss'].tolist())
            ep_best = scores_df.loc[(scores_df['val_loss'] == 
                                     best_val)]['epoch_n'].tolist()[0]
            if ep_best != best_ep:
                print('Best recorded model is actually epoch '
                       '{}, but actual best model is {} based on dataframe. '
                       'Using dataframe one.').format(warn_str, ep_best, 
                                                      ep_df)
            ep_info = scores_df.loc[(scores_df['epoch_n'] == ep_best)]
            if len(ep_info) != 1:
                print(('{} {} lines found in dataframe for epoch '
                        '{}.').format(warn_str, len(ep_info), ep_best))

    return ep_info
    
    
#############################################
def get_scores(dirname='.', alg='sklearn'):
    """
    get_scores()

    Returns line from a saved score dataframe of the max epoch recorded,
    and saved hyperparameter dictionary. 
    
    Prints a warning if no models are recorded or
    the recorded model does not have a score recorded.
    
    Optional args:
        - dirname (str): directory in which scores 'scores_df.csv' and 
                         hyperparameters (hyperparameters.json) are recorded.
                         default: '.'
        - alg (str)    : algorithm used to run logistic regression 
                         ('sklearn' or 'pytorch')
                         default: 'sklearn'
    Returns:
        - ep_info (pd DataFrame): score dataframe line for max epoch recorded.
        - hyperpars (dict)      : dictionary containing hyperparameters
    """

    warn_str='===> Warning: '
    df_path = os.path.join(dirname, 'scores_df.csv')
    
    # get max epoch based on recorded model
    #### WON'T WORK FOR SKLEARN
    if alg == 'pytorch':
        best_ep = get_epoch_n_pt(dirname, 'best')

    # get scores df
    if os.path.exists(df_path):
        scores_df = file_util.loadfile(df_path)
    else:
        print('{} No scores were recorded.'.format(warn_str))
        scores_df = None
        if alg == 'pytorch' and best_ep is not None:
            print(('{} Highest recorded model is for epoch {}, but no '
                    'score is recorded.').format(warn_str, best_ep))

    hyperpars = file_util.loadfile('hyperparameters.json', dirname)

    if alg == 'pytorch':
        # check max epoch recorded matches scores df
        ep_info = check_scores_pt(scores_df, best_ep, hyperpars)
    elif alg == 'sklearn':
        ep_info = scores_df
    else:
        gen_util.accepted_values_error('alg', alg, ['pytorch', 'sklearn'])

    return ep_info, hyperpars


#############################################
class StratifiedShuffleSplitMod(StratifiedShuffleSplit):
    def __init__(self, n_splits=10, train_p=0.75, sample=False, 
                 bal=False, split_test=False, random_state=None):
        """
        Initializes splitting object, which allows classes to be sampled and/or 
        balanced, and the test set to be split into 2.

        Sets attributes:
            - _split_test (bool)     : whether to split the test set
            - _bal (bool)            : whether to balance classes
            - _sample (bool)        : whether to sample class(es)
            if self._sample:
                - _tr_sample (int or list): number of training examples to 
                                            sample, per class if list (for 
                                            class 1 otherwise)
                - _ts_sample (int or list): number of test examples to sample, 
                                            per class if list (for class 1 
                                            otherwise)
            - _set_idx (list)   : list of indices for each set (including the 
                                  extra test set, if self._split_test)

        Optional args:
            - n_splits (int)            : number of splits
                                          default: 10
            - train_p (float)           : training set percentage
                                          default: 0.75
            - sample (int, list or bool): number of values to sample for class 
                                          1 (if int) or per class (if list)
                                          default: False
            - bal (bool)                : if True, classes are balanced
                                          default: False  
            - split_test (bool)         : if True, validation class is split 
                                          into 2 to create a test class 
            - random_state (RandomState): random state or int or None
                                          default: None
        """

        super().__init__(n_splits=n_splits, train_size=train_p, 
                         random_state=random_state)
        self._sample = bool(sample)
        if self._sample:
            if not isinstance(sample, list):
                self._tr_sample = int(sample * train_p)
                self._ts_sample = sample - self._tr_sample
            else:
                self._tr_sample = [int(s * train_p) for s in sample]
                self._ts_sample = [s-t for s,t in zip(sample, self._tr_sample)]
        self._bal     = bal
        self._set_idx = []
        self._split_test = split_test

    def _samp_bal(self, y, train, test):
        """
        self._samp_bal(y, train, test)
        
        Yields subsampled training and test set indices, checking that class 
        sizes are still big enough.

        Required args:
            - y (1D array)    : full array of target values
            - train (1D array): array of data indices for training set
            - test (1D array) : array of data indices for test set
        
        Yields:
            - train (1D array): subsampled array of data indices for training 
                                set
            - test (1D array) : subsampled array of data indices for test set
        """

        sets = [train, test]
        samp_ns = [np.inf, np.inf]
        if self._sample:
            samp_ns = [self._tr_sample, self._ts_sample]        
        for s, (subset, ns) in enumerate(zip(sets, samp_ns)):
            classes, y_indices = np.unique(y[subset], return_inverse=True)
            n_classes = classes.shape[0]
            class_counts = np.bincount(y_indices).tolist()
            class_indices = np.split(np.argsort(y_indices, kind='mergesort'),
                                     np.cumsum(class_counts)[:-1])
            samp_idx = []
            ns = gen_util.list_if_not(ns)
            if self._sample:
                if len(ns) == 1:
                    samp_idx = [1]
                elif len(ns) != n_classes:
                    raise ValueError('If several sample values are provided, '
                                     'should be as many as classes.')
                else:
                    samp_idx = range(n_classes)
            if self._bal:
                # get smallest number
                n = min(class_counts + ns)
                ns = [n for _ in range(n_classes)]
                samp_idx = range(n_classes)
            if min(ns) < n_classes:
                raise ValueError('The smallest set sizes = {} should be '
                                 'greater or equal to the number of '
                                 'classes = {}'.format(min(ns), n_classes))
            sub_idx = []
            for i in range(n_classes):
                if i in samp_idx:
                    class_indices[i] = class_indices[i][ :ns[samp_idx.index(i)]]
                sub_idx.extend(class_indices[i])
            sub_idx = np.sort(sub_idx)
            sets[s] = subset[sub_idx]
        train, test = sets

        return train, test


    def _split_test_set(self, test, y):
        """
        self._split_test_set(test, y)

        Returns split test set, checking that class sizes are still big enough.

        Required args:
            - test (1D array): array of test set indices
            - y (1D array)   : full array of target values

        Returns
            - test     (1D array): array of test set indices
            - test_out (1D array): array of additional test set indices

        """
        classes, y_indices = np.unique(y[test], return_inverse=True)
        class_counts = np.bincount(y_indices)
        n_classes = len(classes)
        if min(class_counts)//2 < n_classes:
            raise ValueError('Cannot split test set into 2 as smallest set '
                    'size = {} should be greater or equal to the number of '
                    'classes = {}'.format(min(class_counts)//2, n_classes))
        test, test_out = [], []
        for counts, idx in zip(class_counts, y_indices):
            test.extend(idx[:counts//2])
            test_out.extend(idx[counts//2:])
        test = np.sort(test)
        test_out = np.sort(test_out)

        return test, test_out


    def _iter_indices(self, X, y, groups=None):
        """
        self._iter_indices(X, y)

        Iterates through the splits to yield train and test sets, and updates 
        self._set_idx with set indices.

        Required args:
            - X (nd array): data array, where first dimension is the trials
            - y (1D array): target array
        
        Optional args:
            - groups (object): ignored, exists for compatibility

        Yields:
            - train (1D array): array of data indices for training set
            - test (1D array) : array of data indices for test set
        """

        for train, test in super()._iter_indices(X, y, groups=groups):   
            if self._sample or self._bal:
                train, test = self._samp_bal(y, train, test)
            if self._split_test:
                test, test_out = self._split_test_set(test, y)
                self._set_idx.append([train, test, test_out])
            else:
                self._set_idx.append([train, test])
            yield train, test


#############################################
class ModData(TransformerMixin):
    def __init__(self, scale=True, extrem=False, shuffle=False, **kwargs):
        """
        Initializes a data modification tool to flatten, optionally scales
        using RobustScaler/MinMaxScaler and optionally shuffle input data.

        Sets attributes:
            - _orig_shape (tuple)    : original shape, as (frames, channels), 
                                       though set to None
            - _scaler (scaler)       : scaler to use (None if none)
            - _extrem (bool)         : if True, 5/95th percentiles are used to 
                                       MinMax scale data
            - _shuffle (bool)        : shuffle boolean

        Optional args:
            - scale (bool)  : if True, data is scaled by channel
            - extrem (bool) : if True, 5/95th percentiles are used to 
                              MinMax scale data
            - shuffle (bool): if True, X is shuffled
        """
        if scale:
            # self._scaler = MinMaxScaler(copy=True, **kwargs)
            # self._extrem = extrem

            # using different quantile range
            self._scaler = RobustScaler(copy=True, quantile_range=(5.0, 95.0), 
                                        **kwargs)
            self._extrem = False # do not use with RobustScaler
        else:
            self._scaler = None
            self._extrem = False
        self._shuffle = shuffle
        self._orig_shape = None
    
    def fit(self, X, y=None, **kwargs):
        """
        Fits original shape and scaler. Runs only on train set.

        Sets attribute:
            - self._orig_shape (tuple): original shape, as (frames, channels)
        
        Calls:
            - self._scaler.fit(): scaler fitter

        Required args:
            - X (3D array): data array, structured as trials x frames x channels
        
        Optional args:
            - y (1D array): target array (ignored)

        Returns:
            - self (TransformerMixin): self
        """

        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        if self._scaler is not None:
            X = self._flatten(X, across='tr')
            if self._extrem:
                X = math_util.extrem_to_med(X, ext_p=[5, 95])
            self._scaler.fit(X, **kwargs)
        return self

    def fit_transform(self, X, y=None, **kwargs):
        """
        Fits original shape and scaler, and returns optionally scaled, shuffled 
        input data, flattened across channels.
        Runs only on train set.

        Calls:
            - self.fit()
            if self._scaler:
                - self._get_scaled()
            if self._shuffle:
                - self._get_shuffled()
            - self._flatten()

        Required args:
            - X (3D array): data array, structured as trials x frames x channels
        
        Optional args:
            - y (1D array): target array (ignored)
        """
        
        X = np.array(X)
    
        self.fit(X, y, **kwargs)
        if self._scaler is not None:
            X = self._get_scaled(X, **kwargs)
        X = self._flatten(X, across='ch')
        if self._shuffle:
            X = self._get_shuffled(X)
        return X

    def transform(self, X, flatten=True, training=False, **kwargs):
        """
        Returns data, optionally scaled, and flattened across channels. Runs 
        only on non train set.

        Calls:
            if self._scaler:
                - self._get_scaled()
            - self._flatten()

        Required args:
            - X (3D array): data array, structured as trials x frames x channels

        Optional args:
            - flatten (bool) : if True, array is flattened across channels
                               (default applies during predict steps)
                               default: True
            - training (bool): if True, training modifications (e.g., shuffle) 
                               are applied
                               (default applies during predict steps)
                               default: False
        
        Returns:
            - X (2D array): data array, structured as trials x (frames/channels)
        """

        X = np.array(X)
        if self._scaler is not None:
            X = self._get_scaled(X)
        if flatten:
            X = self._flatten(X, across='ch')
        if training and self._shuffle:
            X = self._get_shuffled(X)
        return X

    def _flatten(self, X, across='ch'):
        """
        Returns data flattened across channels and frames or trials and frames.

        Required args:
            - X (3D array): data array, structured as trials x frames x channels

        Optional args:
            - across (str): how to flatten data, i.e. channels ('ch') or 
                            across trials ('tr')
                            default: 'ch'
        Returns:
            - X (2D array): data array, structured as:
                if across == 'ch':
                    trials x frames/channels
                elif across == 'tr':
                     trials/frames x channels

        """
        # Reshape X to <= 2 dimensions
        if len(X.shape) == 3: 
            if across == 'ch':
                n_dims = np.prod(self._orig_shape)
                X = X.reshape(-1, n_dims)
            elif across == 'tr':
                n_dims = np.prod([X.shape[0], self._orig_shape[0]])
                X = X.reshape(n_dims, -1)
            else:
                gen_util.accepted_values_error('across', across, ['ch', 'tr'])
        elif len(X.shape) > 3:
            raise ValueError('X should have max 3 dimensions.')
        return X

    def _reshape(self, X):
        """
        Returns X in its original shape.

        Required args:
            - X (nd array): data array

        Returns:
            - X (3D array): data array, structured as trials x frames x channels
        """
        # Reshape X back to its original shape
        if len(X.shape) >= 2:
            X = X.reshape([-1, * self._orig_shape])
        return X

    def _get_scaled(self, X, **kwargs):
        """
        Returns X scaled by each channel.

        Required args:
            - X (3D array): data array, structured as trials x frames x channels

        Returns:
            - X (3D array): data array, scaled by each channel, 
                            structured as trials x frames x channels
        """
        X = self._flatten(X, across='tr')
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _get_shuffled(self, X):
        """
        Returns X shuffled across trials. Creates a new shuffling index if none
        exists and uses a previously created one otherwise.

        Sets attributes:
            - _shuff_reidx (1D array): corresponding shuffling index for targets 

        Required args:
            - X (nd array): data array, structured as trials x rest

        Returns:
            - X (nd array): data array, shuffled across trials, structured as 
                                trials x rest
        """

        if not hasattr(self, '_shuff_reidx'):
            idx = np.asarray(range(len(X))) # get trial indices
            np.random.shuffle(idx)
            # to get sort index corresponding to targets, not input
            self._shuff_reidx = np.argsort(idx)
        else:
            # reconstitute train sort idx from target sort idx
            idx = np.argsort(self._shuff_reidx)
        X = X[idx]
        return X


#############################################
def run_logreg_cv_sk(input_data, targ_data, logregpar, extrapar, 
                     scale=True, sample=False, split_test=False, 
                     parallel=False, max_size=9e7):
    """
    run_logreg_cv_sk(roi_seqs, seq_classes, logregpar, extrapar)

    Runs all runs of logistic regression using sklearn and returns 
    models, crossvalidation split object and extra parameters. Allows saves
    the model under 'models.sav'

    Required args:
        - input_data (3D array) : trace array, structured as 
                                      trials x frames x ROIs
        - seq_classes (2D array): target classes, structured as class values x 1
        - logregpar (LogRegPar) : named tuple containing logistic regression 
                                  parameters
        - extrapar (dict)       : dictionary with extra parameters
            ['epochs'] (int)  : max number of epochs
            ['n_runs'] (int)  : number of runs (split) to run
            ['shuffle'] (bool): if True, data is shuffled
    
    Optional args:
        - scale (bool)          : if True, data is scaled by ROI during training
                                  default: True
        - sample (int or list)  : number of values to sample (if list applies 
                                  to all classes, otherwise applies to class 1)
                                  default: False
        - split_test (bool)     : if True, test sets are split in half to 
                                  create a 'test_out'
                                  default: False
        - parallel (bool)       : if True, splits are run in parallel
                                  default: False

    Returns:
        - mod_cvs (dict)   : cross-validation dictionary with keys:
            ['estimator'] (list)     : list of fitted estimator pipelines for 
                                       each split
            ['fit_time'] (1D array)  : array of fit times for each split
            ['score_time'] (1D array): array of test score times for reach split
            for all combinations of sets ('train', 'test') and 
                scores ('neg_log_loss', 'accuracy', 'balanced_accuracy'):
            ['set_score'] (list)     : array of scores for each split
        - cv (Split object): StratifiedShuffleSplitMod object
        - extrapar (dict)  : dictionary with extra parameters
            ['scoring'] (list)     : sklearn names of scores used
            ['loss_name'] (str)    : name of the loss function used
            ['shuffle'] (bool)     : if True, data is shuffled
    """
    
    n_jobs = gen_util.get_n_jobs(extrapar['n_runs'], parallel=parallel)

    # modify n_jobs if input_data size is too big
    rat = np.prod(input_data.shape)/max_size
    if rat > 1:
        if n_jobs is not None:
            n_jobs = int(n_jobs/rat)
            if n_jobs in [0, 1]:
                n_jobs = None
        if n_jobs is None:
            print(('OOM error possibly upcoming as input data '
                   'size is {}.').format(np.prod(input_data.shape)))

    extrapar = copy.deepcopy(extrapar)
    extrapar['loss_name'] = 'Weighted BCE loss with L2 reg'
    extrapar['scoring'] = ['neg_log_loss', 'accuracy', 'balanced_accuracy']

    mod = LogisticRegression(C=5, fit_intercept=True, class_weight='balanced', 
                             penalty='l2', solver='lbfgs', 
                             max_iter=logregpar.n_epochs)
    scaler = ModData(scale=scale, extrem=True, shuffle=extrapar['shuffle'])
    cv = StratifiedShuffleSplitMod(n_splits=extrapar['n_runs'], 
                   train_p=logregpar.train_p, sample=sample, 
                   bal=logregpar.bal, split_test=split_test)

    mod_pip = make_pipeline(scaler, mod)

    mod_cvs = cross_validate(mod_pip, input_data, targ_data, cv=cv, 
                             return_estimator=True, return_train_score=True, 
                             n_jobs=n_jobs, verbose=3, 
                             scoring=extrapar['scoring'])

    print('Training done.\n')

    # Save models
    fullname = os.path.join(extrapar['dirname'], 'models.sav')
    with open(fullname, 'wb') as f:
        pkl.dump(mod_cvs, f)    

    return mod_cvs, cv, extrapar


#############################################
def test_logreg_cv_sk(mod_cvs, cv, scoring, main_data=None, extra_data=None, 
                      extra_name=None, extra_cv=None):
    """
    test_logreg_cv_sk(mod_cvs, cv, scoring)

    Tests sklearn logistic regression on additional datasets.

    Required args:
        - mod_cvs (dict)   : cross-validation dictionary with keys:
            ['estimator'] (list): list of fitted estimator pipelines for 
                                  each split
        - cv (Split object): StratifiedShuffleSplitMod object
        - scoring (list)   : sklearn names of scores to use
    
    Optional args:
        - main_data (list)       : list of main [input data, target data], 
                                   with input data structured as 
                                       seq x frames x channels 
                                   and target data structured as 
                                       class values x 1.
                                   Only used if the test dataset was split.
                                   default: None
        - extra_data (list)      : list of extra [input data, target data], 
                                   with input data structured as 
                                       seq x frames x channels 
                                   and target data structured as 
                                       class values x 1
                                   default: None
        - extra_name (str)       : name for extra dataset (required if 
                                   extra_data is not None)
                                   default: None
        - extra_cv (Split object): StratifiedShuffleSplitMod object for extra
                                   dataset
                                   default: None

    Returns:
        - mod_cvs (dict): cross-validation dictionary with keys:
            ['estimator'] (list)    : list of fitted estimator pipelines for 
                                      each split
            for all combinations of sets ('test_out' and extra_name) and 
                scores (e.g., 'neg_log_loss', 'accuracy', 'balanced_accuracy'):
            ['set_score'] (1D array): array of scores for each split
    """

    mod_cvs = copy.deepcopy(mod_cvs)
    split_test = cv._split_test

    all_tests, all_data = [], []
    if split_test:
        if main_data is None:
            raise ValueError('If testing additional test set, must provide '
                             '`main_data`.')
        all_tests.append('test_out')
    
    if extra_data is not None:
        if extra_name is None or extra_cv is None:
            raise ValueError('If providing extra data to test set, must '
                             'provide `extra_name` and extra_cv.')
        all_tests.append(extra_name)
        splitter = extra_cv.split(extra_data[0], extra_data[1])
        

    for score in scoring:
        for test in all_tests:
            mod_cvs['{}_{}'.format(test, score)] = \
                np.empty(len(mod_cvs['estimator'])) * np.nan

    for m, mod in enumerate(mod_cvs['estimator']):
        all_data = []
        if split_test:
            idx = cv._set_idx[m][2] # retrieve test_out indices
            all_data.append([main_data[0][idx], main_data[1][idx]])
        if extra_data is not None:
            all_idx = next(splitter)
            idx = [i for sub in all_idx for i in sub]
            all_data.append([extra_data[0][idx], extra_data[1][idx]])
        for score in scoring:
            for test, data in zip(all_tests, all_data):
                key = '{}_{}'.format(test, score)
                sc = get_scorer(score)
                mod_cvs[key][m] = sc(mod, data[0], data[1])

    return mod_cvs


#############################################
def get_transf_data_sk(mod, data, flatten=False, training=False):
    """
    get_transf_data_sk(mod, data):

    Returns data transformed as in model pipeline, using ModData transformation.

    Required args:
        - mod (Pipeline) : model pipeline
        - data (3D array): input array, structured as seqs x frames x channels

    Optional args:
        - flatten (bool) : if True, array is flattened across channels
                           default: False
        - training (bool): if True, training data modifications are applied 
                           (e.g., shuffling)
                           default: False
    
    Returns:
        - transf_data (2-3D array): transformed input array, structured as 
                                    seqs x frames(/channels) (x channels)
    """

    transf_data = mod['moddata'].transform(data, flatten=flatten, 
                                           training=training)
    return transf_data


#############################################
def create_score_df_sk(mod_cvs, saved_idx, set_names, scoring):
    """
    create_score_df_sk(mod_cvs, saved_idx, set_names, scoring)

    Returns scores dataframe from the models provided, specified set names
    and scores.

    Required args:
        - mod_cvs (dict)  : cross-validation dictionary with keys:
            ['estimator'] (list): list of fitted estimator pipelines for 
                                  each split
            for all combinations of sets (set_names) and scores (scoring):
            ['set_score'] (list): array of scores for each split
        - saved_idx (int) : index of the best model
        - set_names (list): set names ('train', 'test', etc.)
        - scoring (list)  : score names ('neg_log_loss', 'accuracy', etc.)

    Returns:
        - scores (pd DataFrame): scores dataframe with columns 
                                 ('run_n', 'n_epochs', and abbreviated 
                                 combinations of set_names and scores) 
    """

    sc_modif = []
    sc_sign = []
    for sc_name in scoring:
        sign = 1
        if 'neg_log_loss' in sc_name:
            sc_name = 'loss'
            sign = -1
        else:
            sc_name = sc_name.replace('accuracy', 'acc').replace('balanced', 
                                      'bal')
            if 'acc' in sc_name:
                sign = 100
            if sc_name == 'bal_acc':
                sc_name = 'acc_bal'
        sc_modif.append(sc_name)
        sc_sign.append(sign)

    set_sc = ['{}_{}'.format(st_name, sc_name) for st_name in set_names 
                                               for sc_name in sc_modif]
    scores = pd.DataFrame(columns=['run_n', 'epoch_n'] + set_sc)

    scores['run_n'] = range(len(mod_cvs['estimator']))
    scores['saved'] = 0
    scores.loc[saved_idx, 'saved'] = 1
    for set_name in set_names:
        for score, sc_mod, sign in zip(scoring, sc_modif, sc_sign):
            key = '{}_{}'.format(set_name, score)
            if key in mod_cvs.keys():
                sc = mod_cvs[key]
                scores['{}_{}'.format(set_name, sc_mod)] = sc * sign
            else:
                print('{} score missing.'.format(key))

    for r in range(len(mod_cvs['estimator'])):
        epoch_n = mod_cvs['estimator'][r]['logisticregression'].n_iter_[0]
        scores.loc[r, 'epoch_n'] = epoch_n

    return scores

