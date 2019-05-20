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
            - acc_class0 (float): number of class 0 samples correctly predicted
            - acc_class1 (float): number of class 1 samples correctly predicted
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
def print_loss(s, loss):
    """
    print_loss(s, loss)

    Print loss for set.
    
    Required args:
        - s (str)     : set (e.g., 'train')
        - loss (float): loss score
    """

    print('    {} loss: {:.4f}'.format(s, loss))


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
def fit_model(info, n_epochs, mod, dls, device, dirname='.', ep_freq=50, 
              test_dl2_name=None):
    """
    fit_model(info, epochs, mod, dls, device)

    Fits model to data and evaluates. Returns an array of scores and an array
    recording which epochs models were saved for.
    
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

    Returns:
        - scores (pd DataFrame): dataframe in which scores are recorded, with
                                 columns epoch_n, saved_eps and combinations of
                                 sets x score types
    """

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

    scores = pd.DataFrame()
    scores = pd.DataFrame(np.nan, index=list(range(n_epochs)), columns=col_names)
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
            print('Epoch {}'.format(ep))
            print_loss('train', scores.loc[ep_loc]['train_loss'].tolist()[0])
            print_loss('val', scores.loc[ep_loc]['val_loss'].tolist()[0])
    
    return scores


#############################################
def get_epoch_n(dirname, model='best'):
    """
    get_epoch_n(dirname)

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
def load_params(dirname, model='best'):
    """
    load_params(dirname)

    Returns model parameters: epoch number, model weights and model biases. 
    Expects models to be recorded as 'ep*.pth', where the digits in the name 
    specify the epoch number. 
    
    Required args:
        - dirname (str): directory path

    Optional args:
        - model (str): model to return ('best', 'first' or 'last')
                       default: 'best'

    Returns:
        if recorded models are found:
            - ep (int)           : number of the requested epoch 
            - weights (2D Tensor): LogReg network weights, 
                                   structured as 1 x n input layer nodes
            - weights (1D Tensor): LogReg network bias, single value Tensor
                                   otherwise returns None
    """

    ep = get_epoch_n(dirname, model)
    if model == 'best':
        ext_str = '_best'

    if ep is None:
        return None
    else:
        models = glob.glob(os.path.join(dirname, 'ep{}*.pth'.format(ep)))[0]
        checkpoint = torch.load(models)
        weights = checkpoint['net']['lin.weight']
        biases = checkpoint['net']['lin.bias']
        return ep, weights, biases


#############################################
def load_checkpoint(mod, filename):
    """
    load_checkpoint(filename)

    Returns model updated with recorded parameters and optimizer state. 
    
    Required args:
        - filename (str)       : name of the file (should be '.pth')
        - mod (torch.nn.Module): Neural network module with optimizer as 
                                 attribute

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
    plot_weights(ax, mod_params, xran)

    Plots weights by frame on axis.
    
    Required args:
        - ax (plt Axis subplot): subplot
        - mod_params (list)    : model parameters [weights, bias]
        - xran (1D array)      : array of x range values

    Optional args:
        - stats (str)      : stats to take, i.e., 'mean' or 'median'
                             default: 'mean'
        - error (str)      : error to take, i.e., 'std' (for std or quintiles) 
                             or 'sem' (for SEM or MAD)
                             default: 'std'
    """

    weights = np.reshape(np.asarray(mod_params[1]), (len(xran), -1))
    wei_stats = math_util.get_stats(weights, stats, error, axes=1)
    title = 'Model weights (ep {})'.format(mod_params[0])
    plot_util.plot_traces(ax, xran, wei_stats[0], wei_stats[1:], title, 
                          col='dimgrey', alpha=0.4)
    ax.axhline(y=0, ls='dashed', c='k', lw=1, alpha=0.5)


#############################################
def get_stats(tr_data, tr_classes, classes, len_s, stats='mean', error='sem'):
    """
    plot_tr_data(tr_data, tr_classes, classes, len_s)

    Plots training data and returns figure, data subplot and trace colors.
    
    Required args:
        - tr_stats (nd array)  : training data array, structured as 
                                 trials x frames x units
        - tr_classes (1D array): training data targets
        - classes (list)       : list of class names
        - len_s (float)        : length of x axis in seconds

    Optional args:
        - stats (str): stats to take, i.e., 'mean' or 'median'
                       default: 'mean'
        - error (str): error to take, i.e., 'std' (for std or quintiles) or 
                       'sem' (for SEM or MAD)
                       default: 'std
    Returns:
        - xran (1D array)     : x values for frames
        - all_stats (3D array): training statistics, structured as 
                                   class x stats (me, err) x frames
        - ns (list)           : number of sequences per class
    """

    xran = np.linspace(0, len_s, tr_data.shape[1])

    ns = []
    # select class trials and take the stats across trials (axis=0), 
    # then across e.g., cells (last axis)
    all_stats = []
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
                 plot_wei=True, stats='mean', error='sem', modeldir='.', 
                 cols=None, data_type=None, xlabel=None):
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
        - fig (plt fig)        : pyplot figure to plot on. If fig or ax_data is
                                 None, new ones are created.
                                 default: None
        - ax_data (plt Axis)   : pyplot axis subplot to plot data on. If fig or 
                                 ax_data is None, new ones are created.
                                 default: None
        - plot_wei (bool)      : if True, weights are also plotted, if a model
                                 was recorded and no fig or ax_data is passed.
                                 default: True
        - stats (str)          : stats to take, i.e., 'mean' or 'median'
                                 default: 'mean'
        - error (str)          : error to take, i.e., 'std' (for std or 
                                 quintiles) or 'sem' (for SEM or MAD)
                                 default: 'std
        - dirname (str)        : name of the directory from which to load
                                 model parameters
                                 default: '.'
        - cols (list)          : colors to use
                                 default: None 
        - data_type (str)      : data type if not training (e.g., test)
                                 default: None
        - xlabel (str)         : x axis label
                                 default: None
    
    Returns:
        - fig (plt fig)                : pyplot figure
        - ax_data (pyplot Axis subplot): subplot
        - cols (list)                  : list of trace colors
    """

    if fig is None or ax_data is None:
        # training data: trials x frames x units
        mod_params = load_params(modeldir, 'best')
        if plot_wei and mod_params is not None:
            fig, ax = plt.subplots(2, figsize=(8, 8), sharex=True, 
                                   gridspec_kw = {'height_ratios':[3, 1]})
            ax_data = ax[0]
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
        plot_weights(ax[1], mod_params, xran, stats, error)
        ax_data.set_xlabel('') # remove redundant label
    
    if xlabel is not None:
        if plot_wei and mod_params is not None:
            ax[1].set_xlabel(xlabel)
        else:
            ax_data.set_xlabel(xlabel)

    
    return fig, ax_data, cols


#############################################
def plot_scores(scores, classes, loss_name='loss', dirname='.', gen_title=''):

    """
    plot_scores(epochs, scores, classes)

    Plots each score type in a figure and saves figures.
    
    Required args:
        - scores (pd DataFrame): dataframe in which scores are recorded, for
                                 each epoch
        - classes (list)       : list of class names
    
    Optional args:
        - loss_name (str)    : name of type of loss
                               default: 'loss'
        - dirname (str)      : name of the directory in which to save figure
                               default: '.'
        - gen_title (str)    : general plot titles
                               default: ''
    """

    epochs = list(range(min(scores['epoch_n']), max(scores['epoch_n']) + 1))

    sc_labs = get_sc_types('label')
    set_labs, set_names = [], []
    for col_name in scores.keys():
        if sc_labs[0] in col_name:
            set_labs.append(col_name.replace('_{}'.format(sc_labs[0]), ''))
            set_names.append('{} set'.format(set_labs[-1]))

    sc_titles = get_sc_names(loss_name, classes) # for title

    for sc_title, sc_lab in zip(sc_titles, sc_labs):
        fig, ax = plt.subplots(figsize=[20, 5])
        for set_lab in set_labs:
            dashes = (None, None)
            if set_lab == 'train':
                dashes = [3, 2]
            if set_lab == 'val':
                dashes = [6, 2]
            sc = np.asarray(scores['{}_{}'.format(set_lab, sc_lab)])
            ax.plot(epochs, sc, label=set_lab ,lw=2.5, dashes=dashes)
            ax.set_title(u'{}\n{}'.format(gen_title, sc_title))
            ax.set_xlabel('Epochs')
        if 'acc' in sc_lab:
            ax.set_ylim(-5, 105)
        elif 'loss' in sc_lab:
            ax.set_ylim(0, ax.get_ylim()[-1]*1.15)
        ax.legend()
        fig.savefig(os.path.join(dirname, '{}'.format(sc_lab)))


#############################################
def check_scores(scores_df, best_ep, hyperpars):
    """
    check_scores(scores_df, best_ep, hyperpars)

    Returns data for the best epoch recorded in scores dataframe, checking that 
    the best epoch in dataframe (based on validation loss) is also the best 
    epoch model saved.
    
    Required args:
        - scores_df (pd DataFrame): scores dataframe
        - best_ep (int)           : max epoch recorded
        - hyperpars (dict)        : dictionary containing hyperparameters

    Returns:
        - ep_info (pd DataFrame): line from score dataframe of max epoch 
                                  recorded.
    """

    warn_str='===> Warning: '

    ep_info = None

    if scores_df is not None:
        # check that all epochs were recorded and correct epoch
        # was recorded as having lowest validation loss
        ep_rec = scores_df.count(axis=0)
        if min(ep_rec) < hyperpars['logregpar']['epochs']:
            print(('{} Only {} epochs were fully '
                    'recorded.').format(warn_str, min(ep_rec)))
        if max(ep_rec) > hyperpars['logregpar']['epochs']:
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
                       'Using dataframe one.').format(warn_str, max_ep, 
                                                      max_ep_df)
            ep_info = scores_df.loc[(scores_df['epoch_n'] == ep_best)]
            if len(ep_info) != 1:
                print(('{} {} lines found in dataframe for epoch '
                        '{}.').format(warn_str, len(ep_info), ep_best))

    return ep_info
    
    
#############################################
def get_scores(dirname='.'):
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

    Returns:
        - ep_info (pd DataFrame): score dataframe line for max epoch recorded.
        - hyperpars (dict)      : dictionary containing hyperparameters
    """

    warn_str='===> Warning: '
    df_path = os.path.join(dirname, 'scores_df.csv')
    
    # get max epoch based on recorded model
    best_ep = get_epoch_n(dirname, 'best')

    # get scores df
    if os.path.exists(df_path):
        scores_df = file_util.loadfile(df_path)
    else:
        print('{} No scores were recorded.'.format(warn_str))
        scores_df = None
        if best_ep is not None:
            print(('{} Highest recorded model is for epoch {}, but no '
                    'score is recorded.').format(warn_str, best_ep))
    
    hyperpars = file_util.loadfile('hyperparameters.json', dirname)

    # check max epoch recorded matches scores df
    ep_info = check_scores(scores_df, best_ep, hyperpars)


    return ep_info, hyperpars

