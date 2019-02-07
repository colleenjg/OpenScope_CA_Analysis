"""
logreg_util.py

Functions and classes for logistic regressions.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 2.7.

"""
import os
import glob
import re

import torch
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import file_util, gen_util, math_util, plot_util


#############################################
class LogReg(torch.nn.Module):
    """
    The LogReg object is a pytorch Neural Network module object that 
    implements a logistic regression.
    """
    
    def __init__(self, num_units, num_steps):
        """
        self.__init__(num_units, num_steps)

        Creates the new LogReg object using the specified 2D input dimensions
        which are flattened to form a 1D input layer. 
        
        The network is composed of a single linear layer from the input layer 
        to a single output on which a sigmoid is applied.

        Initializes num_units, num_steps, lin and sig attributes.

        Required arguments:
            - num_units (int): nbr of units.
            - num_steps (int): nbr of steps per unit.
        """

        super(LogReg, self).__init__()
        self.num_units = num_units
        self.num_steps = num_steps
        self.lin = torch.nn.Linear(self.num_units*self.num_steps, 1)
        self.sig = torch.nn.Sigmoid()
        
    def forward(self, x):
        return self.sig(self.lin(x.view(-1, self.num_units*self.num_steps))).view(-1, 1)


#############################################        
class weighted_BCE():
    """
    The weighted_BCE object defines a weighted binary crossentropy (BCE) loss.
    """
    def __init__(self, weights=None):
        """
        self.__init__()

        Creates the new weighted_BCE object using the specified weights. 
        
        Initializes weights, name attributes.

        Optional arguments:
            - weights (list): list of weights for both classes [class0, class1] 
                              default: None
        """

        if weights is not None and len(weights) != 2:
            raise IOError('Exactly 2 weights must be provided, if any.')
        
        self.weights = weights
        self.name = 'Weighted BCE loss'

    def calc(self, pred_class, act_class):
        """
        self.calc(pred_class, act_class)

        Calculates the weighted BCE loss between the predicted and actual 
        classes using the weights.
        
        Required arguments:
            - pred_class (nd torch Tensor): array of predicted classes (0 and 1s)
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
def class_weights(train_classes):
    """
    class_weights(train_classes)

    Calculates the weights for classes based on their proportions in the
    training class as: train_len/(n_classes * n_class_values).
    
    Required arguments:
        - train_class (nd array) : array of training classes

    Returns:
        - weights (list): list of weights for each classe

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

    Calculates the accuracy for both classes, and returns the actual number of 
    samples for each class as well as the accuracy.
    
    Required arguments:
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

    act_class  = np.asarray(act_class)
    pred_class = np.round(np.asarray(pred_class))
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
def get_sc_types(info='label', ds_len=None):
    """
    get_sc_types()

    Returns info about the four score types: either labels, titles, list 
    indices, or lists to track scores within an epoch.
    
    Optional arguments:
        - info (str)  : type of info to return (label, title, idx or track)
                        default: 'label'
        - ds_len (int): dataset length, needed if info is 'track'
                        default: None

    Returns:
        if info == 'label':
            - label (list): list of score type labels
        elif info == 'title':
            - title (list): list of score type titles
        elif info == 'idx':
            - idx (dict)  : dictionary with index of each score type label
        elif info == 'track':
            - ep_sc (list): list to record epoch scores
            - divs (list) : list to record number of samples to divide by
            - mult (list) : list of values to multiply by to get % for 
                            accuracies
    """

    label = ['loss', 'acc', 'acc_class0', 'acc_class1']
    
    if info == 'label':
        return label

    elif info == 'title':
        title = ['Loss', 'Accuracy (%)', 'Accuracy on class0 trials (%)', 
                 'Accuracy on class1 trials (%)']
        return title

    elif info == 'idx':
        idx = dict()
        for i, lab in enumerate(label):
            idx[lab] = i

        return idx
    
    elif info == 'track':
        if ds_len is None:
            raise ValueError(('If \'info\' is \'track\', must provide ' 
                              '\'ds_len\'.'))
        ep_sc = [0, 0, 0, 0] # loss, acc, acc_class0 and acc_class1
        divs = [ds_len, ds_len, 0, 0] 
        mult = [1., 100., 100., 100.] # to get % values for accuracies

        return ep_sc, divs, mult


#############################################
def get_sc_names(loss_name, classes):
    """
    get_sc_names(loss_name, classes)

    Returns specific score names, incorporating name of type of loss function
    and the class names.
    
    Required arguments:
        - loss_name (str): name of loss function
        - classes (list) : list of names of each class

    Returns:
        - sc_names (list): list of specific score names
    """

    sc_names = get_sc_types(info='title')
    for i, sc_name in enumerate(sc_names):
        if sc_name in ['Loss', 'loss']:
            sc_names[i] = loss_name
        for j, class_name in enumerate(classes):
            generic = 'class{}'.format(j)
            if generic in sc_name:
                sc_names[i] = sc_name.replace(generic, class_name)
    
    return sc_names


#############################################
def get_set_labs(test=True):
    """
    get_set_labs()

    Returns labels for each set (train, val, test).
    
    Optional arguments:
        - test (bool): if True, a test set is included
                       default: True

    Returns:
        - sets (list): list of set labels
    """

    sets = ['train', 'val']
    if test:
        sets.extend(['test'])
    
    return sets


#############################################
def get_sc_labs(test=True, by='flat'):
    """
    get_sc_labs()

    Returns labels for each set (train, val, test).
    
    Optional arguments:
        - test (bool): if True, a test set is included
                       default: True
        - by (str)   : if 'flat', labels are returned flat. If 'set', labels
                       are returned by set.
                       default: 'flat'

    Returns:
        - sc_labs (list): list of set and score labels, 
                          nested by set if by is 'set'.
    """

    sets = get_set_labs(test)
    scores = get_sc_types()
    if by == 'flat':
        sc_labs = ['{}_{}'.format(s, sc) for s in sets for sc in scores]
    elif by == 'set':
        sc_labs = [['{}_{}'.format(s, sc) for sc in scores] for s in sets]
    else:
        gen_util.accepted_values_error('by', by, ['flat', 'set'])

    return sc_labs


#############################################
def run_batches(mod, dl, device, ep_sc, divs, mult, train=True):
    """
    run_batches(mod, dl, device, ep_sc, divs, mult)

    Run dataloader batches through network and returns scores.
    
    Required arguments:
        - mod (torch.nn.Module): Neural network module with optimizer and loss 
                                 function as attributes
        - dl (torch DataLoader): Dataloader
        - device (str)         : device to use ('cuda' or 'cpu') 
        - ep_sc (list)         : list to record epoch scores
        - divs (list)          : list to record number of samples to divide by
        - mult (list)          : list of values to multiply by to get % for 
                                 accuracies

    Optional arguments:
        - train (bool): if True, network is trained on data. If False, 
                        network is evaluated on data, but not trained.
                        default: True

    Returns:
        - ep_sc (list): list of epoch scores
    """

    sc_idx = get_sc_types('idx')
    acc_idx = [sc_idx['acc_class0'], sc_idx['acc_class1']]

    for _, (data, targ) in enumerate(dl, 0):
        if train:
            mod.opt.zero_grad()
        pred_class = mod(data.to(device))
        loss = mod.loss_fct.calc(pred_class, targ.to(device))
        if train:
            loss.backward()
            mod.opt.step()
        ep_sc[sc_idx['loss']] += loss.item()*len(data) # retrieve sum across batch
        ns, accs = accuracy(pred_class.cpu().detach(), targ.cpu().detach())
        ep_sc[sc_idx['acc']] += accs[0] + accs[1]
        for i, n, acc in zip(acc_idx, ns, accs):
            if acc is not None:
                ep_sc[i] += acc
                divs[i] += float(n)
    
    for i in range(len(ep_sc)):
        ep_sc[i] = ep_sc[i] * mult[i]/float(divs[i])
    
    return ep_sc


#############################################
def run_dl(mod, dl, device, train=True):
    """
    run_dl(mod, dl, device)

    Sets model to train or evaluate and runs dataloader through network and 
    returns scores.
    
    Required arguments:
        - mod (torch.nn.Module): Neural network module with optimizer and loss 
                                 function as attributes
        - dl (torch DataLoader): Dataloader
        - device (str)         : device to use ('cuda' or 'cpu') 

    Optional arguments:
        - train (bool): if True, network is trained on data. If False, 
                        network is evaluated on data, but not trained.
                        default: True

    Returns:
        - ep_sc (list): list of epoch scores
    """

    if train:
        mod.train()
    else:
        mod.eval()

    ds_len = dl.dataset.n_samples
    ep_sc, divs, mult = get_sc_types(info='track', ds_len=ds_len)
    
    if train:
        ep_sc = run_batches(mod, dl, device, ep_sc, divs, mult, train)
    else:
        with torch.no_grad():
            ep_sc = run_batches(mod, dl, device, ep_sc, divs, mult, train)
    
    return ep_sc


#############################################
def print_loss(s, loss):
    """
    print_loss(s, loss)

    Print loss for set.
    
    Required arguments:
        - s (str)     : set (e.g., 'train')
        - loss (float): loss score
    """

    print('    {} loss: {}'.format(s, loss))


#############################################
def save_model(info, ep, mod, scores, dirname='.', rem_prev=True): 
    """
    save_model(info, ep, mod, scores)

    Saves model and optimizer, as well as a dictionary with info and epoch 
    scores.
    
    Required arguments:
        - info (dict)          : dictionary of info to save along with model
        - ep (int)             : epoch number
        - mod (torch.nn.Module): Neural network module with optimizer as 
                                 attribute
        - scores (nested list) : list of scores structured as set x score

    Optional arguments:
        - dirname (str)  : directory in which to save
                           default: '.'
        - rem_prev (bool): if True, only the current epoch model and 
                           dictionary are kept, and previous ones are deleted.
                           default: True
    """

    if rem_prev:
        # delete previous model
        prev_model = glob.glob(os.path.join(dirname, 'ep*.pth'))
        prev_json = glob.glob(os.path.join(dirname, 'ep*.json'))
        
        if len(prev_model) == 1 and len(prev_json) == 1:
            os.remove(prev_model[0])
            os.remove(prev_json[0])

    savename = 'ep{}'.format(ep)
    savefile = os.path.join('{}'.format(dirname), savename)
    
    torch.save({'net': mod.state_dict(), 'opt': mod.opt.state_dict()},
               '{}.pth'.format(savefile))
    
    info['epoch'] = ep

    test = len(scores) - 2
    sc_labs = get_sc_labs(test=test, by='set')

    for set_labs, set_scores in zip(sc_labs, scores):
        for sc_lab, score in zip(set_labs, set_scores):
            info[sc_lab] = score

    file_util.save_info(info, savename, dirname, 'json')
    

#############################################
def fit_model(info, epochs, mod, dls, device, dirname='.', ep_freq=50, 
              keep='best'):
    """
    fit_model(info, epochs, mod, dls, device)

    Fits model to data and evaluates. Returns an array of scores and an array
    recording which epochs models were saved for.
    
    Required arguments:
        - info (dict)          : dictionary of info to save along with model
        - epochs (int)         : total number of epochs
        - mod (torch.nn.Module): Neural network module with optimizer and loss 
                                 function as attributes
        - dls (list)           : list of Torch Dataloaders
        - device (str)         : device to use ('cuda' or 'cpu') 

    Optional arguments:
        - dirname (str): directory in which to save models and dictionaries
                         default: '.'
        - ep_freq (int): frequency at which to print loss to console
                         default: 50
        - keep (str)   : if 'best', only the best model is saved. Otherwise,
                         each best to date model is kept.

    Returns:
        - scores (3D array)   : array in which scores are recorded, structured
                                as epochs x nbr sets x nbr score types
        - saved_eps (2D array): array recording which epochs models are 
                                saved for, structured as epochs x 1
    """

    test   = len(dls) - 2
    sets   = get_set_labs(test=test)
    tr_idx, val_idx = [sets.index(x) for x in ['train', 'val']]

    scs = get_sc_types('label')
    loss_idx = scs.index('loss')

    # ep x set (train, val, test) x sc (loss, acc, acc0, acc1)
    scores = np.empty([epochs, len(sets), len(scs)])*np.nan

    min_val = np.inf # value to beat to start recording models
    saved_eps = np.zeros([epochs, 1])

    for ep in range(epochs):
        for i, (s, dl) in enumerate(zip(sets, dls)):
            if ep !=0 and s == 'train':
                # First train epoch: record untrained model
                train = True
            else:
                train = False
            scores[ep, i, :] = run_dl(mod, dl, device, train=train)

        # record model if training is lower than val, and val reaches a new low
        if (scores[ep, tr_idx, loss_idx]*0.99 < scores[ep, val_idx, loss_idx] and 
            scores[ep, val_idx, loss_idx] < min_val):
            if keep == 'best':
                # reset to 0s
                saved_eps = np.zeros([epochs, 1])
                rem_prev = True
            else:
                rem_prev = False
            save_model(info, ep, mod, scores[ep], dirname, rem_prev)
            min_val = scores[ep, val_idx, loss_idx]
            saved_eps[ep] = 1

        if ep==0 or ep%ep_freq == 0:
            print('Epoch {}'.format(ep))
            print_loss('train', scores[ep, tr_idx, loss_idx])
            print_loss('val', scores[ep, val_idx, loss_idx])
        
    return scores, saved_eps


#############################################
def get_max_epoch(dirname):
    """
    get_max_epoch(dirname)

    Returns max epoch recorded in a directory.
    
    Required arguments:
        - dirname (str): name of the directory

    Returns:
        - max_ep (int): number of max epoch recorded 
    """

    warn_str='===> Warning: '
    models = glob.glob(os.path.join(dirname, 'ep*.pth'))
    if len(models) > 0:
        max_ep = max([int(re.findall(r'\d+', os.path.split(mod)[-1])[0]) 
                     for mod in models])
    else:
        max_ep = None
        print('{} No models were recorded.'.format(warn_str))

    return max_ep


#############################################
def load_params(dirname):
    """
    load_params(dirname)

    Loads model parameters: max epoch, model weights and model biases. 
    
    Required arguments:
        - dirname (str): name of the directory

    Returns:
        - max_ep (int)       : number of max epoch recorded 
        - weights (2D Tensor): LogReg network weights, 
                               structured as 1 x n input layer nodes
        - weights (1D Tensor): LogReg network bias, single value Tensor
    """

    max_ep = get_max_epoch(dirname)
    
    if max_ep is None:
        return None
    else:
        models = os.path.join(dirname, 'ep{}.pth'.format(max_ep))
        checkpoint = torch.load(models)
        weights = checkpoint['net']['lin.weight']
        biases = checkpoint['net']['lin.bias']
        return max_ep, weights, biases


#############################################
def load_checkpoint(mod, filename):
    """
    load_checkpoint(filename)

    Loads model and optimizer state from recorded file. 
    
    Required arguments:
        - filename (str)       : name of the file (should be '.pth')
        - mod (torch.nn.Module): Neural network module with optimizer as 
                                 attribute

    Returns:
        - mod (torch.nn.Module): Neural network module with model and optimizer
                                 updated.
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
        raise IOError('No checkpoint found at \'{}\''.format(checkpt_name))

    return mod


#############################################
def get_stats(xran, data, stats='mean', error='sem'):
    """
    get_stats(xran, data)

    Get statistics (e.g., mean + SEM) across trials, then units. Returns
    statistics as a chunk array.
    
    Required arguments:
        - xran (1D array): array of x range values
        - data (nd array): data array, structured as trials x steps x units

    Optional arguments:
        - stats (str)    : stats to take, i.e., 'mean' or 'median'
                           default: 'mean'
        - error (str)    : error to take, i.e., 'std' (for std or quintiles) or 
                           'sem' (for SEM or MAD)
                           default: 'std'

    Returns:
        - data_stats (2D array): stats array, structured as 
                                 stat type (xran, me, error) x steps
    """

    # take the mean/median across trials
    trial_me = math_util.mean_med(data, axis=0, stats=stats)

    # mean/med along units axis (last)
    me = math_util.mean_med(trial_me, stats=stats, axis=-1) 
    err = math_util.error_stat(trial_me, stats=stats, error=error, axis=-1)
    
    if not(stats=='median' and error=='std'):
        data_stats = np.stack([xran, me, err])
    else:
        data_stats = np.concatenate([xran[np.newaxis, :], me[np.newaxis, :], 
                                    np.asarray(err)], axis=0)
    return data_stats


#############################################
def plot_weights(ax, mod_params, xran, stats='mean', error='sem'):
    """
    plot_weights(ax, mod_params, xran)

    Plots weights by step on axis.
    
    Required arguments:
        - ax (plt Axis subplot): subplot
        - mod_params (list)    : model parameters [weights, bias]
        - xran (1D array)      : array of x range values

    Optional arguments:
        - stats (str)      : stats to take, i.e., 'mean' or 'median'
                             default: 'mean'
        - error (str)      : error to take, i.e., 'std' (for std or quintiles) or 
                             'sem' (for SEM or MAD)
                             default: 'std'
    """

    weights = np.reshape(np.asarray(mod_params[1]), (1, len(xran), -1))
    wei_stats = get_stats(xran, weights, stats, error)
    plot_util.plot_traces(ax, wei_stats, stats=stats, error=error, 
                          col='dimgrey', alpha=0.4, fluor=None)
    ax.axhline(y=0, ls='dashed', c='k', lw=1, alpha=0.5)
    ax.set_title('Model weights (ep {})'.format(mod_params[0]))


#############################################
def plot_tr_data(tr_data, tr_classes, classes, len_s, plot_wei=True, 
                 dirname='.', stats='mean', error='sem'):
    """
    plot_tr_data(tr_data, tr_classes, classes, len_s)

    Plots training data.
    
    Required arguments:
        - tr_data (nd array)   : training data array, structured as 
                                 trials x steps x units
        - tr_classes (1D array): training data targets
        - classes (list)       : list of class names
        - len_s (float)        : length of full step range in seconds
        - plot_wei (bool)      : if True, weights are also plotted, if a model
                                 was recorded
    
    Optional arguments:
        - dirname (str)        : name of the directory in which to save figure
                                 default: '.'
        - stats (str)          : stats to take, i.e., 'mean' or 'median'
                                 default: 'mean'
        - error (str)          : error to take, i.e., 'std' (for std or 
                                 quintiles) or 'sem' (for SEM or MAD)
                                 default: 'std'
    
    Returns:
        - fig (plt fig)                : pyplot figure
        - ax_data (pyplot Axis subplot): subplot
        - cols (list)                  : list of colors for plotting
    """

    # training data: trials x steps x units
    cols = ['steelblue', 'coral']
    mod_params = load_params(dirname)

    if plot_wei and mod_params is not None:
        fig, ax = plt.subplots(2, figsize=(8, 8), sharex=True, 
                                  gridspec_kw = {'height_ratios':[3, 1]})
        ax_data = ax[0]
    else:
        fig, ax_data = plt.subplots()

    xran = np.linspace(0, len_s, tr_data.shape[1])

    for i, class_name in enumerate(classes):
        # select class trials and take the stats across trials (axis=0), 
        # then across e.g., cells (last axis)
        idx = (tr_classes == i) # bool array
        n = sum(idx)
        class_stats = get_stats(xran, tr_data[idx], stats, error)
        leg = '{} (n={})'.format(class_name, n)
        plot_util.plot_traces(ax_data, class_stats, stats=stats, 
                              error=error, col=cols[i], alpha=0.8/len(classes), 
                              label=leg, fluor=None)

    # plot weights as well
    if plot_wei and mod_params is not None:
        plot_weights(ax[1], mod_params, xran, stats, error)
        ax_data.set_xlabel('') # remove redundant label
    
    return fig, ax_data, cols


#############################################
def plot_scores(epochs, scores, classes, dirname='.', loss_name='loss', 
                test=True, gen_title='', fig_ext='.svg'):

    """
    plot_scores(epochs, scores, classes)

    Plots each score type in a figure and saves each figure.
    
    Required arguments:
        - epochs (int)     : nbr of epochs to plot
        - scores (3D array): array in which scores are recorded, structured
                             as epochs x nbr sets x nbr score types
        - classes (list)   : list of class names
    
    Optional arguments:
        - dirname (str)  : name of the directory in which to save figure
                           default: '.'
        - loss_name (str): name of type of loss
                           default: 'loss'
        - test (bool)    : if True, test set scores are also provided
                           default: True
        - gen_title (str): general plot titles
                           default: ''
        - fig_ext (str)  : extension for saving figure
                           default: '.svg'
    """

    sc_titles = get_sc_names(loss_name, classes) # for title
    set_labs = ['{} set'.format(x) for x in get_set_labs(test=test)] # for legend
    sc_labs = get_sc_labs(test=test) # for file name

    cols = ['lightsteelblue', 'cornflowerblue', 'royalblue']  

    for i, [sc_title, sc_lab] in enumerate(zip(sc_titles, sc_labs)):
        fig, ax = plt.subplots(figsize=[20, 5])
        for j, [set_lab, col] in enumerate(zip(set_labs, cols)):
            ax.plot(range(epochs), scores[:, j, i], label=set_lab, color=col)
            ax.set_title('{}\n{}'.format(gen_title, sc_title))
            ax.set_xlabel('Epochs')
        ax.legend()
        fig.savefig(os.path.join(dirname, '{}{}'.format(sc_lab, fig_ext)), 
                    bbox_inches='tight')


#############################################
def save_scores(df_labs, df_info, epochs, saved_eps, scores, test=True, 
                dirname='.'):
    """
    save_scores(df_labs, df_info, epochs, saved_eps, scores)

    Saves run information and scores per epoch as a dataframe.
    
    Required arguments:
        - df_labs             : basic dataframe labels 
                                (e.g., mouse_n, sess_n, etc.)
        - df_info             : dictionary with run info for df_labs
        - epochs (int)        : nbr of epochs to plot
        - saved_eps (2D array): array recording which epochs models are 
                                saved for, structured as epochs x 1
        - scores (3D array)   : array in which scores are recorded, structured
                                as epochs x nbr sets x nbr score types

    Optional arguments:
        - test (bool)    : if True, test set scores are also provided
                           default: True
        - dirname (str)  : name of the directory in which to save figure
                           default: '.'
    """
    
    sc_labs = get_sc_labs(test=test)
    all_labels = df_labs + ['epoch', 'saved'] + sc_labs
    df_vals = np.asarray([np.asarray([df_info[key]]*epochs) 
                          for key in all_labels if key in df_info.keys()]).T
    epoch_arr = np.asarray(range(epochs))[:, np.newaxis]
    
    scores = np.reshape(scores, [scores.shape[0], -1]) # flatten
    summ_data = np.concatenate([df_vals, epoch_arr, saved_eps, scores], axis=1)
    summ_df = pd.DataFrame(data=summ_data, columns=all_labels)

    file_util.save_info(summ_df, 'scores_df', dirname, 'csv')


#############################################
def check_scores(scores_df, max_ep, hyperpars):
    """
    check_scores(scores_df, max_ep, hyperpars)

    Loads saved scores and returns data for max epoch, checking that the max
    epoch in dataframe is also the max epoch model saved.
    
    Required arguments:
        - scores_df (pd DataFrame): scores dataframe
        - max_ep (int)            : max epoch recorded
        - hyperpars (dict)        : dictionary containing hyperparameters

    Return:
        - ep_info (pd DataFrame): score dataframe line for max epoch recorded.
    """

    warn_str='===> Warning: '

    ep_info = None

    if scores_df is not None:
        # check that all epochs were recorded and correct epoch
        # was recorded as having lowest validation loss
        ep_rec = scores_df.count(axis=0)
        if min(ep_rec) < hyperpars['epochs']:
            print(('{} Only {} epochs were fully '
                    'recorded.').format(warn_str, min(ep_rec)))
        if max(ep_rec) > hyperpars['epochs']:
            print(('{} {} epochs were '
                    'recorded.').format(warn_str, max(ep_rec)))
        if len(scores_df.loc[(scores_df['saved'] == 1)]['epoch'].tolist()) == 0:
            print(('{} No models were recorded in '
                    'dataframe.').format(warn_str))
        else:
            max_ep_df = max(scores_df.loc[(scores_df['saved'] == 1)]['epoch'].tolist())
            if max_ep_df != max_ep:
                print(('{} Highest recorded model is actually epoch '
                       '{}, but expected {} based on dataframe. Using '
                       'dataframe one.').format(warn_str, max_ep, max_ep_df))
            ep_info = scores_df.loc[(scores_df['epoch'] == max_ep_df)]
            if len(ep_info) != 1:
                print(('{} {} lines found in dataframe for epoch '
                        '{}.').format(warn_str, len(ep_info), max_ep_df))

    return ep_info
    
    
#############################################
def get_scores(direc):
    """
    get_scores(direc)

    Loads saved scores and dictionary and returns data for max epoch and 
    hyperparameter dictionary. 
    
    Prints a warning if no models are recorded or
    the recorded model does not have a score recorded.
    
    Required arguments:
        - direc (str): directory in which scores are recorded

    Return:
        - ep_info (pd DataFrame): score dataframe line for max epoch recorded.
        - hyperpars (dict)      : dictionary containing hyperparameters
    """

    warn_str='===> Warning: '
    df_path = os.path.join(direc, 'scores_df.csv')
    
    # get max epoch based on recorded model
    max_ep = get_max_epoch(direc)

    # get scores df
    if os.path.exists(df_path):
        scores_df = file_util.load_file(df_path, file_type='csv')
    else:
        print('{} No scores were recorded.'.format(warn_str))
        scores_df = None
        if max_ep is not None:
            print(('{} Highest recorded model is for epoch {}, but no '
                    'score is recorded.').format(warn_str, max_ep))
    
    hyperpars = file_util.load_file('hyperparameters.json', direc, 
                                    file_type='json')

    # check max epoch recorded matches scores df
    ep_info = check_scores(scores_df, max_ep, hyperpars)


    return ep_info, hyperpars

