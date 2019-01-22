import yaml
import os
import argparse
import random
import glob
import re
import multiprocessing

from matplotlib import pyplot as plt
import scipy.stats as st
import torch
import torchvision
import h5py
import pickle
import pandas as pd
from joblib import Parallel, delayed

np = torch._np

from util import file_util, gen_util, str_util, math_util, plot_util

#############################################
def seed_all(seed, device='cpu', print_seed=True):

    if seed is None:
        seed = random.randint(1, 10000)
        if print_seed:
            print('Random seed: {}'.format(args.seed))
    else:
        if print_seed:
            print('Preset seed: {}'.format(args.seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    return seed


#############################################
def get_runname(mouse_n, sess_n, layer, raw=False, norm=False, comp='surp', 
                shuffle=False):

    fluor_str = str_util.fluor_par_str(raw, type_str='file')
    norm_str = str_util.norm_par_str(norm, type_str='file')
    shuff_str = str_util.shuff_par_str(shuffle, type_str='file')

    runname = 'm{}_s{}_{}_{}{}_{}{}'.format(mouse_n, sess_n, layer, fluor_str, norm_str, comp, shuff_str)

    return runname


#############################################
def get_rundirec_dict(rundirec):

    # format: mouse_sess_layer_fluor_norm_comp_shuffled/uniqueid_run
    parts = os.path.split(rundirec)
    first = parts[0].split('_')
    second = parts[1].split('_')
    if len(second) == 3:
        # rejoin if uniqueid is datetime
        second = ['{}_{}'.format(second[0], second[1]), int(second[2])]
    else:
        second = [int(x) for x in second] 

    if first[4] == 'norm':
        norm = True
        comp = first[5]
    else:
        norm = False
        comp = first[4]
    
    if first[-1] == 'shuffled':
        shuffle = True
    else:
        shuffle = False

    rundirec_dict = {'mouse_n': int(first[0][1]),
                     'sess_n':int(first[1][1]),
                     'layer': first[2],
                     'fluor': first[3],
                     'norm': norm,
                     'comp': comp,
                     'shuffled': shuffle,
                     'uniqueid': second[0],
                     'run': int(second[1])
                     }
    
    return rundirec_dict


#############################################
def create_datasets(roi_tr_segs, seg_classes, args, test=True):
    
    # ordered as trials x frames x ROIs

    # number of segs in each set
    if test:
        valtest_n = int(len(roi_tr_segs)*(1.0-args.train_p)/2)
        valtest = np.random.choice(range(len(roi_tr_segs)), (2, valtest_n), replace=False)
        val_idx = sorted(valtest[0])
        test_idx = sorted(valtest[1])

    else:
        val_n = int(len(roi_tr_segs)*(1.0-args.train_p))
        val_idx = sorted(np.random.choice(range(len(roi_tr_segs)), val_n, replace=False))
        test_idx = list() # empty
    
    train_idx = sorted(set(range(len(roi_tr_segs))) - set(val_idx) - set(test_idx))

    if args.shuffle:
        np.random.shuffle(seg_classes)

    train_data  = torch.Tensor(roi_tr_segs[train_idx])
    train_class = torch.Tensor(seg_classes[train_idx])
    val_data    = torch.Tensor(roi_tr_segs[val_idx])
    val_class   = torch.Tensor(seg_classes[val_idx])

    if args.norm:
        all_tr_flatter = train_data.view((-1,) + train_data.size()[2:])
        train_means = torch.mean(all_tr_flatter, dim=0)
        train_stds = torch.std(all_tr_flatter, dim=0)
        train_data = (train_data - train_means)/train_stds
        val_data = (val_data - train_means)/train_stds
        args.train_means = train_means.tolist()
        args.train_stds = train_stds.tolist()

    if test:
        test_data   = torch.Tensor(roi_tr_segs[test_idx])
        test_class  = torch.Tensor(seg_classes[test_idx])
        if args.norm:
            test_data = (test_data - train_means)/train_stds
        return train_data, train_class, val_data, val_class, test_data, test_class
    
    else:
        return train_data, train_class, val_data, val_class


#############################################
def info_dict(args, epoch=None):

    if args is not None:
        info = {'mouse_n': args.mouse_n,
                'sess_n': args.sess_n,
                'layer': args.layer,
                'line': args.line,
                'fluor': str_util.fluor_par_str(args.raw, 'file'),
                'norm': args.norm,
                'shuffled': args.shuffle,
                'comp': args.comp,
                'uniqueid': args.uniqueid,
                'run': args.run,
                'runtype': args.runtype,
                'n_roi': args.n_roi,
                }
        
        if epoch is not None:
            info['epoch'] = epoch

    # if no args are passed, just returns keys
    else:
        info = ['mouse_n', 'sess_n', 'layer', 'line', 'fluor', 'norm', 
                'shuffled', 'comp', 'uniqueid', 'run', 'runtype', 'n_roi', 
                'epoch']

    return info


#############################################
def df_cols():

    data_labs = ['train', 'val', 'test']
    sc_names = ['loss', 'acc', 'acc_class0', 'acc_class1']
    sc_labs = ['{}_{}'.format(data, sc) for data in data_labs for sc in sc_names]

    df_labs = info_dict(None) # get keys

    all_labs = df_labs + sc_labs + ['saved']

    return all_labs


#############################################
def summ_cols(CI=95):

    qs = [(100.-CI)*0.5, CI*0.5+50.] # high and lo quartiles
    q_names = []
    for q in qs:
        q_res = q%1
        if q_res == 0:
            q_names.append('q{}'.format(int(q)))
        else:
            q_names.append('q{}p{}'.format(int(q), str(q_res)[2]))
    
    data_labs = ['train', 'val', 'test']
    sc_names = ['loss', 'acc', 'acc_class0', 'acc_class1']
    data_labs = ['epochs'] + ['{}_{}'.format(data, sc) for data in data_labs for sc in sc_names]
    
    stat_names = ['mean', 'std', 'sem', 'med', 'q25', 'q75', 'mad'] + q_names
    stat_labs = ['{}_{}'.format(data, stat) for data in data_labs for stat in stat_names]

    df_labs = gen_util.remove_if(info_dict(None), ['run', 'epoch', 'unique_id'])

    all_labs = df_labs + ['runs_total', 'runs_nan'] + stat_labs

    return all_labs, df_labs, data_labs

#############################################
def get_roi_traces(sess_dict, data_dir, raw=False):
    """
    Load basic information about ROI dF/F traces: number of ROIs, their names, 
    and number of data points in the traces.
    """

    traces_dir = os.path.join(data_dir, sess_dict['traces_dir'])
    with h5py.File(traces_dir, 'r') as f:
        # get names of rois
        roi_names = f['roi_names'].value.tolist()
        # get number of rois
        nroi = len(roi_names)
        # get number of data points in traces
        nframes = f['data'].shape[1]

    if raw:
        roi_tr_file = traces_dir
    else:
        roi_tr_file = os.path.join(data_dir, sess_dict['dff_traces_dir'])
    with h5py.File(roi_tr_file,'r') as f:
        # get traces
        roi_traces = np.asarray(f['data'].value)
    
    return roi_names, nroi, nframes, roi_traces


#############################################
def get_sess_data(data_dir, mouse_n, sess_n, layer, gab_fr=0, comp='surp', 
                  runtype='prod', raw=False):
    
    sess_dict_name = 'sess_dict_mouse{}_sess{}_{}.json'.format(mouse_n, 
                    sess_n, layer)
    sess_dict_dir = os.path.join('session_dicts', runtype)

    if os.path.exists(sess_dict_dir):
        sess_dict = file_util.load_file(sess_dict_name, sess_dict_dir, 'json')

    else:
        print('{} dictionary does not exist.'.format(sess_dict_dir))
        exit()
    
    _, nroi, _, roi_traces = get_roi_traces(sess_dict, data_dir, raw)
    fps = sess_dict['twop_fps']
    frames = sess_dict['frames']
    frame_names = ['A', 'B', 'C', 'D/E']

    if comp == 'surp':
        classes = ['nosurp', 'surp']
        pre = (sess_dict['gab_fr'][0] - gab_fr) *0.3 # in sec
        post = 1.5 - pre # in sec
        all_fr_segs = np.asarray([range(x-int(pre*fps), 
                                  x+int(post*fps)) for x in frames])
        seg_classes = np.zeros([len(frames), 1])
        seg_classes[sess_dict['surp_idx']] = 1
        roi_tr_segs = roi_traces[:, all_fr_segs].transpose(1, 2, 0) # ordered as trials x frames x ROIs
        gabs = frame_names[gab_fr]

    elif comp in['AvB', 'AvC', 'BvC']:
        if comp == 'AvB':
            gab_fr = [0, 1]
            classes = ['gabA', 'gabB'] # was ['Gabor A', 'Gabor B'] before
        elif comp == 'AvC':
            gab_fr = [0, 2]
            classes = ['gabA', 'gabC']
        elif comp == 'BvC':
            gab_fr = [1, 2]
            classes = ['gabB', 'gabC']
        pre = [(sess_dict['gab_fr'][0] - gf)*0.3 for gf in gab_fr] # in sec
        post = [0.45 - p for p in pre] # in sec
        all_fr_segs = [np.asarray([range(x-int(pr*fps), x+int(po*fps)) for x in frames]) for [pr, po] in zip(pre, post)]
        # select segs (92.5% of class 1 segs, 7.5% of class 2 segs) and trim to same length
        class0_segs = np.random.choice(range(len(frames)), int(len(frames)*0.925), replace=False)
        class1_segs = np.random.choice(range(len(frames)), int(len(frames)*0.075), replace=False)
        seg_len = min(all_fr_segs[0].shape[1], all_fr_segs[1].shape[1])
        all_fr_segs = np.concatenate([all_fr_segs[0][class0_segs, :seg_len], 
                                      all_fr_segs[1][class1_segs, :seg_len]], axis=0)
        seg_classes = np.concatenate([np.zeros(len(class0_segs)), np.ones(len(class1_segs))], axis=0)[:, np.newaxis]
        gabs = [frame_names[gf] for gf in gab_fr]
    else:
        gen_util.accepted_values_error('comp', comp, ['surp', 'AvB', 'AvC', 'BvC'])

    roi_tr_segs = roi_traces[:, all_fr_segs].transpose(1, 2, 0) # ordered as trials x frames x ROIs

    if sum(sum(sum(np.isnan(roi_tr_segs)))):
        roi_ok = np.where(sum(sum(np.isnan(roi_tr_segs)))==0)[0]
        nroi = len(roi_ok)
        roi_tr_segs = roi_tr_segs[:, :, roi_ok]
    log_var = np.log(np.var(roi_tr_segs))
    print('Runtype: {}\nMouse: {}\nSess: {}\nLayer: {}\nLine: {}\nROIs: {}\n'
          'Gab fr: {}\nGab K: {}\nFrames per seg: {}\nLogvar: {:.2f}'.format(runtype,
                mouse_n, sess_n, layer, sess_dict['line'], nroi,  
                gabs, sess_dict['gab_k'], roi_tr_segs.shape[1], 
                log_var))
    
    return roi_tr_segs, fps, classes, seg_classes


#############################################
def assemble_chunks(xran, data, stats, error):
    # select class segs and take the mean/median across trials
    data = math_util.mean_med(data, axis=0, stats=stats)
    me = math_util.mean_med(data, stats=stats, axis=1) # mean/med across cells
    err = math_util.error_stat(data, stats=stats, error=error, axis=1)
    if not(stats=='median' and error=='std'):
        chunk_stats = np.stack([xran, me, err])
    else:
        chunk_stats = np.concatenate([xran[np.newaxis, :], me[np.newaxis, :], 
                                      np.asarray(err)], axis=0)
    return chunk_stats


#############################################
def plot_tr_traces(args, data, seg_class):

    cols = ['steelblue', 'coral']
    mod_params = load_params(args.dirname)
    
    if mod_params is None:
        fig_tr, ax_tr = plt.subplots()
    else:
        fig_tr, ax = plt.subplots(2, figsize=(8, 8), sharex=True, 
                                  gridspec_kw = {'height_ratios':[3, 1]})
        ax_tr = ax[0]

    # data: trials x steps x cells
    data = np.asarray(data)
    seg_class  = np.asarray(seg_class).squeeze()

    fluor_str = str_util.fluor_par_str(args.raw, 'print')
    norm_str = str_util.norm_par_str(args.norm, 'print')
    shuff_str = str_util.shuff_par_str(args.shuffle, 'labels')

    if args.comp == 'surp':
        classes = ['nosurp', 'surp']
        xran = np.linspace(0, 1.5, data.shape[1])
        gab_par = {'gab_fr': args.gab_fr, 'pre': 0, 'post': 1.5}
        [xpos, labels_nosurp, h_bars, 
        seg_bars] = plot_util.plot_seg_comp(gab_par, 'nosurp')
        _, labels_surp, _, _ = plot_util.plot_seg_comp(gab_par, 'surp')
        t_heis = [0.85, 0.95]
        labels = [labels_nosurp, labels_surp]

    elif args.comp in ['AvB', 'AvC', 'BvC']:
        xran = np.linspace(0, 0.45, data.shape[1])
        classes = ['Gabor {}'.format(args.comp[i]) for i in [0, 2]]
    else:
        gen_util.accepted_values_error('comp', args.comp, ['surp', 'AvB', 'AvC', 'BvC'])
    
    for i, class_name in enumerate(classes):
        segs = (seg_class == i)
        # select class segs and take the mean/median across trials
        chunk_stats = assemble_chunks(xran, data[segs], args.stats, args.error)
        leg = '{} (n={})'.format(class_name, sum(segs))
        plot_util.plot_traces(ax_tr, chunk_stats, stats=args.stats, 
                              error=args.error, col=cols[i], 
                              alpha=0.8/len(classes), label=leg, raw=args.raw)
        ax_tr.legend()
        ax_tr.set_ylabel('{}{}'.format(fluor_str, norm_str))
        if args.comp == 'surp':
            plot_util.add_labels(ax_tr, labels[i], xpos, t_heis[i], col=cols[i])
    
    if mod_params is not None:
        weights = np.reshape(np.asarray(mod_params[1]), (1, data.shape[1], data.shape[2]))
        chunk_stats = assemble_chunks(xran, weights, args.stats, args.error)
        plot_util.plot_traces(ax[1], chunk_stats, stats=args.stats, 
                              error=args.error, col='dimgrey', alpha=0.4)
        ax[1].axhline(y=0, ls='dashed', c='k', lw=1, alpha=0.5)
        ax[1].set_title('Model weights (ep {})'.format(mod_params[0]))
        ax[1].set_ylabel('')
        ax_tr.set_xlabel('')
    
    if args.comp == 'surp':
        plot_util.add_bars(ax_tr, hbars=h_bars, bars=seg_bars)
    stat_str = str_util.stat_par_str(args.stats, args.error)

    ax_tr.set_title(('Mouse {}, sess {}, {} {}, '
                     '\n{} across ROIs{}').format(args.mouse_n, args.sess_n, 
                                                  args.line, args.layer, 
                                                  stat_str, shuff_str))

    save_name = os.path.join(args.dirname, 'training_traces{}'.format(args.fig_ext))
    fig_tr.savefig(save_name, bbox_inches='tight')


#############################################
def plot_scores(args, scores, classes):

    data_labs = ['train', 'val', 'test']   
    file_names = ['wBCEloss', 'acc', 'acc_{}'.format(classes[0]), 
                  'acc_{}'.format(classes[1])]

    titles = ['Weighted BCE Loss', 'Accuracy (%)', 
              'Accuracy on {} trials (%)'.format(classes[0]), 
              'Accuracy on {} trials (%)'.format(classes[1])]

    cols = ['lightsteelblue', 'cornflowerblue', 'royalblue']  

    fluor_str = str_util.fluor_par_str(args.raw, 'print')
    norm_str = str_util.norm_par_str(args.norm, 'print')
    shuff_str = str_util.shuff_par_str(args.shuffle, 'labels')

    for i, [title, file_name] in enumerate(zip(titles, file_names)):
        fig, ax = plt.subplots(figsize=[20, 5])
        for j, [lab, col] in enumerate(zip(data_labs, cols)):
            ax.plot(range(args.epochs), scores[:, j, i], label=lab, color=col)
            ax.set_title(('Mouse {}, sess {}, {} {}\n'
                          '{}{} ({}{})').format(args.mouse_n, args.sess_n, 
                                                args.line, args.layer, title, 
                                                shuff_str, fluor_str, norm_str))
            ax.set_xlabel('Epochs')
        ax.legend()
        fig.savefig(os.path.join(args.dirname, '{}{}'.format(file_name, 
                    args.fig_ext)), bbox_inches='tight')


#############################################
def save_scores(args, scores, classes, saved_eps):

    # df_info_labels, epoch, scores, saved
    all_labels = gen_util.remove_if(df_cols(), 'epoch')

    scores = np.reshape(scores, [scores.shape[0], -1]) 
    df_info = info_dict(args, None)
    
    df_vals = np.asarray([np.asarray([df_info[key]]*args.epochs) 
                        for key in all_labels if key in df_info.keys()]).T
    
    summ_data = np.concatenate([df_vals, 
                                np.asarray(range(args.epochs))[:, np.newaxis], 
                                scores, saved_eps], axis=1)
    summ_df = pd.DataFrame(data=summ_data, columns=all_labels)

    file_util.save_info(summ_df, 'scores_df', args.dirname, 'csv')


#############################################
def max_epoch(dirname):
    models = glob.glob(os.path.join(dirname, 'ep*.pth'))

    if len(models) > 0:
        max_ep = max([int(re.findall(r'\d+', os.path.split(mod)[-1])[0]) for mod in models])
    else:
        max_ep = None
        print('    Warning: No models were recorded.')
    
    return max_ep


#############################################
def load_params(dirname):

    max_ep = max_epoch(dirname)
    
    if max_ep is None:
        return None
    else:
        models = os.path.join(dirname, 'ep{}.pth'.format(max_ep))
        checkpoint = torch.load(models)
        weights = checkpoint['net']['lin.weight']
        biases = checkpoint['net']['lin.bias']
        return max_ep, weights, biases


#############################################
def init_model_comp(roi_tr_segs, seg_classes, args, test=True):

    all_data = create_datasets(roi_tr_segs, seg_classes, args, test=test)

    train_data, train_class, val_data, val_class = all_data[0:4]
    if test:
        test_data, test_class = all_data[4:6]
    
    weights = compute_weights(train_class)

    model = log_regression(train_data.shape[2], train_data.shape[1]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_dl = torch.utils.data.DataLoader(roi_ds(train_data, train_class), 
                                            batch_size=args.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(roi_ds(val_data, val_class), 
                                            batch_size=args.batch_size)
    if test:
        test_dl = torch.utils.data.DataLoader(roi_ds(test_data, test_class), 
                                                batch_size=args.batch_size)
        return model, optimizer, weights, [train_dl, val_dl, test_dl]
    else:
        return model, optimizer, weights, [train_dl, val_dl]


#############################################
def run_eps(args, mod, opt, wei, dls, test=True):

    # ep x grp (train, val, test) x sc (loss, acc, acc0, acc1)
    scores = np.empty([args.epochs, 3, 4])*np.nan

    min_val = 100 # dummy value to beat to start recording models
    saved_eps = np.zeros([args.epochs, 1])

    norm_str = str_util.norm_par_str(args.norm, 'print')
    shuff_str = str_util.shuff_par_str(args.shuffle, 'labels')

    for ep in range(args.epochs):
        if ep == 0:
            # No training for first epoch
            scores[ep, 0, :] = val(mod, dls[0], wei, args.device)    
        else:
            scores[ep, 0, :] = train(mod, opt, dls[0], wei, args.device)
        scores[ep, 1, :] = val(mod, dls[1], wei, args.device)
        if test:
            scores[ep, 2, :] = val(mod, dls[2], wei, args.device)

        # record model if validation is lower than training, and reaches a new low
        if scores[ep, 1, 0] >= 0.99*scores[ep, 0, 0] and scores[ep, 1, 0] < min_val:
            save_model(ep, args, mod, opt, scores[ep, :, :], test)
            if not args.keep_prev:
                # reset to 0s
                saved_eps = np.zeros([args.epochs, 1])
            min_val = scores[ep, 1, 0]
            saved_eps[ep] = 1

        if ep==0 or (ep)%args.ep_freq == 0:
            print('Run {}{}{}, epoch {}'.format(args.run, norm_str, shuff_str, ep))
            print_loss('train', scores[ep, 0, 0])
            print_loss('val', scores[ep, 1, 0])

    return scores, saved_eps


#############################################
def single_run(roi_tr_segs, seg_classes, classes, args, run):

    args.run = run

    if args.parallel and args.plt_bkend is not None:
        plt.switch_backend(args.plt_bkend) # needs to be repeated within joblib

    if args.uniqueid is None:
        subdir = 'run_{}'.format(run)
    else:
        subdir = '{}_{}'.format(args.uniqueid, run)

    args.dirname = file_util.create_dir([args.output, args.runname, 
                                        subdir], print_dir=False)
    print('Run {} directory: {}\n'.format(run, args.dirname))

    args_dict = args.__dict__

    file_util.save_info(args_dict, 'hyperparameters', args.dirname, 'json')

    mod, opt, wei, dls = init_model_comp(roi_tr_segs, seg_classes, args, 
                                         test=True)

    # scores: ep x grp (train, val, test) x sc (loss, acc, acc0, acc1)
    scores, saved_eps = run_eps(args, mod, opt, wei, dls, test=True)

    print('Run {}: training done.\n'.format(run))

    tr_data = dls[0].dataset.data.numpy()
    tr_class = dls[0].dataset.target.numpy()

    plot_tr_traces(args, tr_data, tr_class)
    
    # save figures of loss, accuracy, etc
    plot_scores(args, scores, classes)
    
    # save scores in dataframe
    save_scores(args, scores, classes, saved_eps)

    plt.close('all')


#############################################
class log_regression(torch.nn.Module):
        def __init__(self, num_units, num_steps):
            super(log_regression, self).__init__()
            self.num_units = num_units
            self.num_steps = num_steps
            self.lin = torch.nn.Linear(self.num_units*self.num_steps, 1)
            self.sig = torch.nn.Sigmoid()
            
        def forward(self, x):
            return self.sig(self.lin(x.view(-1, self.num_units*self.num_steps))).view(-1, 1)


#############################################        
def loss_function(pred_class, act_class, weights=None):
    '''
    weights = [class0, class1]
    '''
    if weights is not None:
        weights = act_class*(weights[1]-weights[0]) + (weights[0])

    BCE = torch.nn.functional.binary_cross_entropy(pred_class, act_class, weight=weights)
    return BCE


#############################################
def accuracy(pred_class, act_class):
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
class roi_ds(torch.utils.data.TensorDataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.n_samples = self.data.shape[0]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        return torch.Tensor(self.data[index]), torch.Tensor(self.target[index])


#############################################
def compute_weights(train_seg_cl):
    # train_len/(n_classes, n_class_values)
    train_seg_cl = np.asarray(train_seg_cl).squeeze()
    classes = list(np.unique(train_seg_cl))
    weights = []
    for cl in classes:
        weights.append((len(train_seg_cl)/(float(len(classes)) *
                        list(train_seg_cl).count(cl))))
    
    return weights


#############################################
def train(mod, optimizer, dl, weights, device):
    mod.train()
    ep_sc = [0, 0, 0, 0] # accuracy, class0 accuracy and class1 accuracy
    ds_len = dl.dataset.n_samples
    divs = [ds_len, ds_len, 0, 0] # count number of examples to divide by
    mult = [1., 100., 100., 100.] # to get % values for accuracies
    for _, (data, targ) in enumerate(dl, 0):
        optimizer.zero_grad()
        pred_class = mod(data.to(device))
        loss = loss_function(pred_class, targ.to(device), weights=weights)
        loss.backward()
        optimizer.step()
        ep_sc[0] += loss.item()*len(data) # retrieve sum across batch
        ns, accs = accuracy(pred_class.cpu().detach(), targ.cpu().detach())
        ep_sc[1] += accs[0] + accs[1]
        for i, (n, acc) in enumerate(zip(ns, accs)):
            if acc is not None:
                ep_sc[i+2] += acc
                divs[i+2] += float(n)
        
    for i in range(len(ep_sc)):
        ep_sc[i] = ep_sc[i]*mult[i]/float(divs[i])
    return ep_sc


#############################################
def val(mod, dl, weights, device):
    mod.eval()   
    ep_sc = [0, 0, 0, 0] # accuracy, class0 accuracy and class1 accuracy
    ds_len = dl.dataset.n_samples
    divs = [ds_len, ds_len, 0, 0] # count number of examples to divide by
    mult = [1., 100., 100., 100.] # to get % values for accuracies
    with torch.no_grad():
        for _, (data, targ) in enumerate(dl, 0):
            pred_class = mod(data.to(device))
            loss = loss_function(pred_class, targ.to(device), weights=weights)
            ep_sc[0] += loss.item()*len(data) # retrieve sum across batch
            ns, accs = accuracy(pred_class.cpu().detach(), targ.cpu().detach())
            ep_sc[1] += accs[0] + accs[1]
            for i, (n, acc) in enumerate(zip(ns, accs)):
                if acc is not None:
                    ep_sc[i+2] += acc
                    divs[i+2] += float(n)

    for i in range(len(ep_sc)):
        ep_sc[i] = ep_sc[i]*mult[i]/float(divs[i])
    
    return ep_sc


#############################################
def print_loss(test_type, loss):
    print('    {} loss: {}'.format(test_type, loss))


#############################################
def save_model(ep, args, mod, opt, scores, test=True): 

    # scores: grp (train, val, test) x sc (loss, acc, acc0, acc1)
    if not args.keep_prev:
        # delete previous model
        prev_model = glob.glob(os.path.join(args.dirname, 'ep*.pth'))
        prev_json = glob.glob(os.path.join(args.dirname, 'ep*.json'))
        
        if len(prev_model) == 1 and len(prev_json) == 1:
            os.remove(prev_model[0])
            os.remove(prev_json[0])

    savename = 'ep{}'.format(ep)
    savefile = os.path.join('{}'.format(args.dirname), savename)
    
    torch.save({'net': mod.state_dict(), 'opt': opt.state_dict()},
               '{}.pth'.format(savefile))
    
    info = info_dict(args, ep)
    datatypes = ['train', 'val']
    if test:
        datatypes.extend(['test'])

    for datatype, score in zip(datatypes, scores):
        info['{}_loss'.format(datatype)] = score[0]
        info['{}_acc'.format(datatype)] = score[1]
        info['{}_acc_class0'.format(datatype)] = score[2]
        info['{}_acc_class1'.format(datatype)] = score[3]
    
    file_util.save_info(info, savename, args.dirname, 'json')


#############################################
def load_checkpoint(filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    checkpt_name = os.path.split(filename)[-1]
    if os.path.isfile(filename):
        print(('\nLoading checkpoint found at \'{}\''.format(checkpt_name)))
        checkpoint = torch.load(filename)
        mod = checkpoint['net']
        opt = checkpoint['opt']
    else:
        raise IOError('No checkpoint found at \'{}\''.format(checkpt_name))

    return mod, opt


#############################################
def plot_my_data(ax, arr, celltype, datatype, mice, sesses, n_rois, analysis, n_runs):
    
    col=['steelblue', 'coral']
    # Session x (n/n rois)           
    if len(mice) == 2:
        n_rois_str = ['{}/{}'.format(int(n_rois[0, 0]), int(n_rois[1, 0])),
                      '{}/{}'.format(int(n_rois[0, 1]), int(n_rois[1, 1]))]
        if len(sesses) == 3:
            n_rois_str.extend(['{}/{}'.format(int(n_rois[0, 2]), int(n_rois[1, 2]))]) 
    else:
        n_rois_str = ['{}'.format(int(n_roi)) for n_roi in list(n_rois.squeeze())]

    sess_labels = ['Session {}\n({} rois)'.format(sess, n_roi) 
                    for [sess, n_roi] in zip(sesses, n_rois_str)]
    for m, mouse in enumerate(mice):
        # non shuffle
        n_runs_m = list(n_runs[m, :, 0].squeeze())
        n_runs_str = '{}/{}'.format(int(n_runs_m[0]), int(n_runs_m[1]))
        if len(sesses) == 3:
            n_runs_str += '/{}'.format(int(n_runs_m[2]))
        ax.errorbar(sess_labels, arr[m, :, 0, 0], yerr=arr[m, :, 0, 1], 
                    fmt='-o', capsize=4, capthick=2, color=col[m],
                    label='mouse {}\n({} runs)'.format(mouse, n_runs_str))
    # shuffle
    shuff_med = np.median(arr[:, :, 1, 0], axis=0)
    shuff_err_lo = np.median(arr[:, :, 1, 1], axis=0)
    shuff_err_hi = np.median(arr[:, :, 1, 2], axis=0)
    n_runs_sh = np.sum(n_runs[:, :, 1], axis=0).squeeze()
    n_runs_str = '{}/{}'.format(int(n_runs_sh[0]), int(n_runs_sh[1]))
    if len(sesses) == 3:
        n_runs_str += '/{}'.format(int(n_runs_sh[2]))

    # plot CI
    ax.bar(sess_labels, height=shuff_err_hi-shuff_err_lo, bottom=shuff_err_lo, 
           color='lightgray', width=0.2, label='shuffled\n({} runs)'.format(n_runs_str))
    # plot median (with some thickness based on ylim)
    y_lim = ax.get_ylim()
    med_th = 0.005*(y_lim[1]-y_lim[0])
    ax.bar(sess_labels, height=med_th, bottom=shuff_med-med_th/2.0, 
           color='grey', width=0.2)

    ax.set_title('{} {} for {} logistic regressions'.format(celltype, datatype, analysis))
    if datatype == 'test_acc':
        ax.set_ylabel('Accuracy (%)')
    elif datatype == 'epochs':
        ax.set_ylabel('Epoch nbr')

    ax.legend()


#############################################
def collate_scores(all_labels, direc, args):
    print(direc)
    
    warn_str = '===> Warning:'
    scores = pd.DataFrame(columns=all_labels)
    models = glob.glob(os.path.join(args.output, direc, 'ep*.pth'))

    # get max epoch number, based on saved models
    if len(models) > 0:
        max_ep = max([int(re.findall(r'\d+', os.path.split(mod)[-1])[0]) 
                        for mod in models])
    else:
        max_ep = None
        print('{} No models were recorded.'.format(warn_str))

    # get hyperparams
    hyperpars = file_util.load_file('hyperparameters.json', 
                                        os.path.join(args.output, direc),
                                        file_type='json')

    df_path = os.path.join(args.output, direc, 'scores_df.csv')

    # get scores df
    if os.path.exists(df_path):
        scores_df = file_util.load_file(df_path, file_type='csv')
    else:
        print('{} No scores were recorded.'.format(warn_str))
        scores_df = None
        if max_ep is not None:
            print(('{} Highest recorded model is for epoch {}, but no '
                    'score is recorded.').format(warn_str, max_ep))
    
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

    if ep_info is None:
        scores.loc[0] = np.empty(scores.shape[1])*np.nan
        runname_dict = get_rundirec_dict(direc)
        runname_dict['runtype'] = hyperpars['runtype']
        runname_dict['line'] = hyperpars['line']
        runname_dict['n_roi'] = hyperpars['n_roi']
        for col in runname_dict.keys():
            scores.loc[0, col] = runname_dict[col]
    else:
        for col in all_labels:
            scores.loc[0, col] = ep_info[col].item()

    return scores
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='logreg_models', help='where to store output')
    parser.add_argument('--task', default='run_regr', help='run_regr or collate')
    parser.add_argument('--comp', default='surp', help='surp, AvB, AvC or BvC')
    parser.add_argument('--fig_ext', default='.svg')
    parser.add_argument('--plt_bkend', default=None, 
                        help='switch matplotlib backend when running on server')

        # run_regr general
    parser.add_argument('--runtype', default='prod', help='prod or pilot')
    parser.add_argument('--mouse_n', default=1, type=int)
    parser.add_argument('--sess_n', default='all')
 
    parser.add_argument('--parallel', action='store_true', 
                        help='do runs in parallel.')
    parser.add_argument('--datadir', default=None, 
                        help=('data directory (if None, uses a directory '
                              'defined below'))
    parser.add_argument('--keep_prev', action='store_true', 
                        help=('keep previous models when a better performing '
                              'model is recorded.'))
        
        # run_regr hyperparameters
    parser.add_argument('--n_reg', default=50, type=int, help='n regular runs')
    parser.add_argument('--n_shuff', default=50, type=int, help='n shuffled runs')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--train_p', default=0.75, type=float, 
                        help='proportion of dataset used in training set')
    parser.add_argument('--stats', default='mean', help='mean or median')
    parser.add_argument('--error', default='sem', help='std or sem')
    parser.add_argument('--gab_fr', default=0, type=int, 
                        help='starting gab frame in comp is surp')
    parser.add_argument('--norm', action='store_true', 
                        help='normalize each ROI trace')
    parser.add_argument('--raw', action='store_true', 
                        help='use raw instead of dF/F ROI traces')
    parser.add_argument('--ep_freq', default=50, type=int,  
                        help='epoch frequency at which to print loss')
    parser.add_argument('--seed', default=None, type=int,  
                        help='manual seed')
    parser.add_argument('--uniqueid', default='datetime', 
                        help=('passed string, \'datetime\' for date and time '
                              'or None for no uniqueid'))

        # analysis parameters
    parser.add_argument('--CI', default=95, type=int, help='shuffled CI')

    args = parser.parse_args()

    # HARD-CODED FOR NOW
    args.device = 'cpu'


    if args.plt_bkend is not None:
        plt.switch_backend(args.plt_bkend) 
    
    if args.task == 'run_regr':
        if args.datadir is None:
            # previously: '/media/colleen/LaCie/CredAssign/pilot_data'
            args.datadir = '../data/AIBS/{}'.format(args.runtype) 

        if args.uniqueid == 'datetime':
            args.uniqueid = str_util.create_time_str()

        mouse_df = file_util.load_file('mouse_df_{}.csv'.format(args.runtype), 
                                       file_type='csv')
        
        if args.sess_n == 'all':
            sesses = mouse_df.loc[(mouse_df['mouseid'] == args.mouse_n) &
                                  (mouse_df['pass_fail'] == 'P') &
                                  (mouse_df['all_files'] == 1)]['overall_sess_n'].tolist()
        else:
            sesses = [int(args.sess_n)]

        for sess_n in sesses:
            args.sess_n = sess_n
            mouse_line = mouse_df.loc[(mouse_df['mouseid'] == args.mouse_n) &
                                      (mouse_df['overall_sess_n'] == args.sess_n)]
            args.layer = mouse_line['layer'].item()
            args.line = mouse_line['line'].item()

            args.seed = seed_all(args.seed, args.device)

            [roi_tr_segs, _, classes, 
                          seg_classes] = get_sess_data(args.datadir, args.mouse_n,  
                                                       args.sess_n, args.layer, 
                                                       comp=args.comp, 
                                                       gab_fr=args.gab_fr, 
                                                       runtype=args.runtype, 
                                                       raw=args.raw)
            args.n_roi = roi_tr_segs.shape[2]

            for runs, shuffle in zip([args.n_reg, args.n_shuff], [False, True]):

                args.shuffle = shuffle
                args.runname = get_runname(args.mouse_n, args.sess_n, args.layer, 
                                           args.raw, args.norm, args.comp, 
                                           args.shuffle)

                if args.parallel:
                    num_cores = multiprocessing.cpu_count()
                    Parallel(n_jobs=num_cores)(delayed(single_run)
                            (roi_tr_segs, seg_classes, classes, args, run) 
                            for run in range(runs))
                else:
                    for run in range(runs):
                        single_run(roi_tr_segs, seg_classes, classes, args, run)

    elif args.task == 'collate':

        subdirs = [os.path.join(subdir, name)
                   for subdir in os.listdir(args.output)
                   if os.path.isdir(os.path.join(args.output, subdir)) 
                      and args.comp in subdir
                   for name in os.listdir(os.path.join(args.output, subdir))]

        all_labels = df_cols()

        if args.parallel:
            num_cores = multiprocessing.cpu_count()
            scores_list = Parallel(n_jobs=num_cores)(delayed(collate_scores)
                          (all_labels, direc, args) for direc in subdirs)
            all_scores = pd.concat(scores_list)
        else:
            all_scores = pd.DataFrame(columns=all_labels)
            for direc in subdirs:
                scores = collate_scores(all_labels, direc, args)
                all_scores = all_scores.append(scores)

        # reorganize by mouse, session, norm, shuffle, uniqueid, run
        sorter = ['mouse_n', 'sess_n', 'fluor', 'norm', 'shuffled', 'uniqueid', 'run']
        all_scores = all_scores.sort_values(by=sorter).reset_index(drop=True)

        file_util.save_info(all_scores, '{}_all_scores_df'.format(args.comp), 
                            args.output, 'csv')

    # analyses accuracy
    elif args.task == 'analyse':
        
        all_scores_df = file_util.load_file('{}_all_scores_df.csv'.format(args.comp),
                                            args.output, 'csv')
        summ_cols, comm_labs, data_labs = summ_cols(args.CI)

        scores_summ = pd.DataFrame(columns=summ_cols)

        # get all mice n
        list_mice = sorted(all_scores_df.mouse_n.unique().tolist())
        for mouse in list_mice:
            # get all sess n
            mouse_lines = all_scores_df.loc[(all_scores_df['mouse_n'] == mouse)]
            list_sess = sorted(mouse_lines.sess_n.unique().tolist())
            for sess in list_sess:
                sess_lines = mouse_lines.loc[(mouse_lines['sess_n'] == sess)]
                list_fluor = sorted(sess_lines.fluor.unique().tolist())
                for fluor in list_fluor:
                    fluor_lines = sess_lines.loc[(sess_lines['fluor'] == fluor)]
                    list_norm = sorted(fluor_lines.norm.unique().tolist())
                    for norm in list_norm:
                        norm_lines = fluor_lines.loc[(fluor_lines['norm'] == norm)]
                        list_shuff = sorted(norm_lines.shuffled.unique().tolist())
                        for shuff in list_shuff:
                            shuff_lines = norm_lines.loc[(norm_lines['shuffled'] == shuff)]
                            curr_lin = len(scores_summ)
                            for lab in comm_labs:
                                # check that all the same
                                lab_vals = shuff_lines[lab].unique().tolist()
                                if len(lab_vals) > 1:
                                    raise ValueError(('Several values found for '
                                                      'mouse {}, sess {}, fluor {},'
                                                      'norm {}, shuff {}')
                                                      .format(mouse, sess, fluor,
                                                              norm, shuff))
                                else:
                                    scores_summ.loc[curr_lin, lab] = lab_vals[0]
                            # calculate runs
                            scores_summ.loc[curr_lin, 'runs_total'] = len(shuff_lines)
                            nan_runs = sum(np.isnan(shuff_lines['epoch'].tolist()))
                            scores_summ.loc[curr_lin, 'runs_nan'] = nan_runs
                            # calculate stats
                            for data_lab in data_labs:
                                


        print('doodle')


                # n_rois = mouse_df.loc[(mouse_df['mouseid'] == int(mouse)) & 
                #                       (mouse_df['overall_sess_n'] == int(sess))]['n_rois'].tolist()[0]
                # layer = sorted(all_scores_df.loc[(all_scores_df['mouse_n'] == mouse) &
                #                                  (all_scores_df['sess_n'] == sess)].layer.unique().tolist())[0]
                # analyses = sorted(all_scores_df.loc[(all_scores_df['mouse_n'] == mouse) &
                #                                     (all_scores_df['sess_n'] == sess)].analysis.unique().tolist())
                # for analys in analyses:
                #     shuffles = sorted(all_scores_df.loc[(all_scores_df['mouse_n'] == mouse) &
                #                                         (all_scores_df['sess_n'] == sess) &
                #                                         (all_scores_df['analysis'] == analys)].shuffle.unique().tolist())
                #     for shuff in shuffles:
                #         def sem_nan(data):
                #             if isinstance(data, list):
                #                 return data
                #             else:
                #                 return [data, np.nan]
                        
                #         if shuff:
                #             stats = 'median'
                #             error = 'std'
                #             qu = [2.5, 97.5]
                #         else:
                #             stats = 'mean'
                #             error = 'sem'
                #             qu = None
                #         list_lines = all_scores_df.loc[(all_scores_df['mouse_n'] == mouse) &
                #                                        (all_scores_df['sess_n'] == sess) &
                #                                        (all_scores_df['analysis'] == analys) &
                #                                        (all_scores_df['shuffle'] == shuff)]

                #         n_runs = len(list_lines) - sum(list_lines['epoch'].isna())
                #         # take epoch sem and mean
                #         epoch_mean = math_util.mean_med(list_lines['epoch'], stats=stats, nanpol='omit')
                #         epoch_sem_qu3, epoch_qu98 = sem_nan(math_util.error_stat(list_lines['epoch'], stats=stats, error=error, nanpol='omit', qu=qu))
                #         # take dff_train_wBCEloss mean and sem
                #         dff_train_wBCEloss_mean = math_util.mean_med(list_lines['dff_train_wBCEloss'], stats=stats, nanpol='omit')
                #         dff_train_wBCEloss_sem_qu3, dff_train_wBCEloss_qu98 = sem_nan(math_util.error_stat(list_lines['dff_train_wBCEloss'], stats=stats, error=error, nanpol='omit', qu=qu))
                #         # take dff_train_acc mean and sem
                #         dff_train_acc_mean = math_util.mean_med(list_lines['dff_train_acc'], stats=stats, nanpol='omit')
                #         dff_train_acc_sem_qu3, dff_train_acc_qu98 = sem_nan(math_util.error_stat(list_lines['dff_train_acc'], stats=stats, error=error, nanpol='omit', qu=qu))
                #         # take dff_train_common_acc mean and sem
                #         dff_train_common_acc_mean = math_util.mean_med(list_lines['dff_train_common_acc'], stats=stats, nanpol='omit')
                #         dff_train_common_acc_sem_qu3, dff_train_common_acc_qu98 = sem_nan(math_util.error_stat(list_lines['dff_train_common_acc'], stats=stats, error=error, nanpol='omit', qu=qu))
                #         # take dff_train_rare_acc mean and sem
                #         dff_train_rare_acc_mean = math_util.mean_med(list_lines['dff_train_rare_acc'], stats=stats, nanpol='omit')
                #         dff_train_rare_acc_sem_qu3, dff_train_rare_acc_qu98 = sem_nan(math_util.error_stat(list_lines['dff_train_rare_acc'], stats=stats, error=error, nanpol='omit', qu=qu))

                #         # take dff_val_wBCEloss mean and sem
                #         dff_val_wBCEloss_mean = math_util.mean_med(list_lines['dff_val_wBCEloss'], stats=stats, nanpol='omit')
                #         dff_val_wBCEloss_sem_qu3, dff_val_wBCEloss_qu98 = sem_nan(math_util.error_stat(list_lines['dff_val_wBCEloss'], stats=stats, error=error, nanpol='omit', qu=qu))
                #         # take dff_val_acc mean and sem
                #         dff_val_acc_mean = math_util.mean_med(list_lines['dff_val_acc'], stats=stats, nanpol='omit')
                #         dff_val_acc_sem_qu3, dff_val_acc_qu98 = sem_nan(math_util.error_stat(list_lines['dff_val_acc'], stats=stats, error=error, nanpol='omit', qu=qu))
                #         # take dff_val_common_acc mean and sem
                #         dff_val_common_acc_mean = math_util.mean_med(list_lines['dff_val_common_acc'], stats=stats, nanpol='omit')
                #         dff_val_common_acc_sem_qu3, dff_val_common_acc_qu98 = sem_nan(math_util.error_stat(list_lines['dff_val_common_acc'], stats=stats, error=error, nanpol='omit', qu=qu))
                #         # take dff_val_rare_acc mean and sem
                #         dff_val_rare_acc_mean = math_util.mean_med(list_lines['dff_val_rare_acc'], stats=stats, nanpol='omit')
                #         dff_val_rare_acc_sem_qu3, dff_val_rare_acc_qu98 = sem_nan(math_util.error_stat(list_lines['dff_val_rare_acc'], stats=stats, error=error, nanpol='omit', qu=qu))

                #         # take dff_test_wBCEloss mean and sem
                #         dff_test_wBCEloss_mean = math_util.mean_med(list_lines['dff_test_wBCEloss'], stats=stats, nanpol='omit')
                #         dff_test_wBCEloss_sem_qu3, dff_test_wBCEloss_qu98 = sem_nan(math_util.error_stat(list_lines['dff_test_wBCEloss'], stats=stats, error=error, nanpol='omit', qu=qu))
                #         # take dff_test_acc mean and sem
                #         dff_test_acc_mean = math_util.mean_med(list_lines['dff_test_acc'], stats=stats, nanpol='omit')
                #         dff_test_acc_sem_qu3, dff_test_acc_qu98 = sem_nan(math_util.error_stat(list_lines['dff_test_acc'], stats=stats, error=error, nanpol='omit', qu=qu))
                #         # take dff_test_common_acc mean and sem
                #         dff_test_common_acc_mean = math_util.mean_med(list_lines['dff_test_common_acc'], stats=stats, nanpol='omit')
                #         dff_test_common_acc_sem_qu3, dff_test_common_acc_qu98 = sem_nan(math_util.error_stat(list_lines['dff_test_common_acc'], stats=stats, error=error, nanpol='omit', qu=qu))
                #         # take dff_test_rare_acc mean and sem
                #         dff_test_rare_acc_mean = math_util.mean_med(list_lines['dff_test_rare_acc'], stats=stats, nanpol='omit')
                #         dff_test_rare_acc_sem_qu3, dff_test_rare_acc_qu98 = sem_nan(math_util.error_stat(list_lines['dff_test_rare_acc'], stats=stats, error=error, nanpol='omit', qu=qu))
                        

                #         all_data = [mouse, sess, layer, line, n_rois, analys, shuff,
                #                     n_runs, epoch_mean, epoch_sem_qu3, epoch_qu98,
                #                     dff_train_wBCEloss_mean, dff_train_wBCEloss_sem_qu3, dff_train_wBCEloss_qu98,
                #                     dff_train_acc_mean, dff_train_acc_sem_qu3, dff_train_acc_qu98, 
                #                     dff_train_common_acc_mean, dff_train_common_acc_sem_qu3, dff_train_common_acc_qu98,
                #                     dff_train_rare_acc_mean, dff_train_rare_acc_sem_qu3, dff_train_rare_acc_qu98,
                #                     dff_val_wBCEloss_mean, dff_val_wBCEloss_sem_qu3, dff_val_wBCEloss_qu98,
                #                     dff_val_acc_mean, dff_val_acc_sem_qu3, dff_val_acc_qu98,
                #                     dff_val_common_acc_mean, dff_val_common_acc_sem_qu3, dff_val_common_acc_qu98,
                #                     dff_val_rare_acc_mean, dff_val_rare_acc_sem_qu3, dff_val_rare_acc_qu98,
                #                     dff_test_wBCEloss_mean, dff_test_wBCEloss_sem_qu3, dff_test_wBCEloss_qu98,
                #                     dff_test_acc_mean, dff_test_acc_sem_qu3, dff_test_acc_qu98,
                #                     dff_test_common_acc_mean, dff_test_common_acc_sem_qu3, dff_test_common_acc_qu98,
                #                     dff_test_rare_acc_mean, dff_test_rare_acc_sem_qu3, dff_test_rare_acc_qu98]
                                               
                #         # add info to full dataframe NOT SAFE... could have wrong columns - improve later
                #         scores_summ.loc[len(scores_summ)] = all_data

                #         # overwrite dataframe at each loop
                #         file_util.save_info(scores_summ, 'summ_scores_df', 
                #                             scores_dir, 'csv')

    elif args.task == 'plot':
        scores_dir = os.path.join(args.output, 'regr_models')
        fluor = 'dff'

        summ_scores_df = file_util.load_file('summ_scores_df.csv', scores_dir, 'csv')
        
        mousetypes = [['1', '3'], ['2'], ['4'], ['6']]
        celltypes = ['L2/3 soma', 'L5 dend', 'L5 soma', 'L2/3 dend']
        celltypes_file = ['L23_soma', 'L5_dend', 'L5_soma', 'L2/3 dend']
        fig_ep_ab, ax_ep_ab = plt.subplots(ncols=2, nrows=2, figsize=(15, 15), sharey=True)
        fig_test_ab, ax_test_ab = plt.subplots(ncols=2, nrows=2, figsize=(15, 15), sharey=True)
        fig_ep_ac, ax_ep_ac = plt.subplots(ncols=2, nrows=2, figsize=(15, 15), sharey=True)
        fig_test_ac, ax_test_ac = plt.subplots(ncols=2, nrows=2, figsize=(15, 15), sharey=True)
        fig_ep_surp, ax_ep_surp = plt.subplots(ncols=2, nrows=2, figsize=(15, 15), sharey=True)
        fig_test_surp, ax_test_surp = plt.subplots(ncols=2, nrows=2, figsize=(15, 15), sharey=True)

        fig_list = [[fig_ep_ab, fig_test_ab], [fig_ep_ac, fig_test_ac], [fig_ep_surp, fig_test_surp]]
        ax_list = [[ax_ep_ab, ax_test_ab], [ax_ep_ac, ax_test_ac], [ax_ep_surp, ax_test_surp]]

        for c, [mice, celltype] in enumerate(zip(mousetypes, celltypes)):
            list_sess = sorted(summ_scores_df.loc[(summ_scores_df['mouse_n'].isin(mice))].sess_n.unique().tolist())
            analyses = sorted(summ_scores_df.loc[(summ_scores_df['mouse_n'].isin(mice))].analysis.unique().tolist())
            for a, analys in enumerate(analyses):
                shuffles = sorted(summ_scores_df.loc[(summ_scores_df['mouse_n'].isin(mice)) &
                                                      (summ_scores_df['analysis']==analys)].shuffle.unique().tolist())
                # create epoch array: mouse x sess x shuffle x stats (mean, sem/3p, sem/98p)
                epoch_arr = np.empty([len(mice), len(list_sess), len(shuffles), 3])
                test_arr = np.empty([len(mice), len(list_sess), len(shuffles), 3])
                n_rois = np.empty([len(mice), len(list_sess)])
                n_runs = np.empty([len(mice), len(list_sess), len(shuffles)])
                for m, mouse in enumerate(mice):
                    for s, sess in enumerate(list_sess):
                        for sh, shuff in enumerate(shuffles):
                            sub_df = summ_scores_df.loc[(summ_scores_df['mouse_n']==mouse) &
                                                        (summ_scores_df['sess_n']==sess) &
                                                        (summ_scores_df['analysis']==analys) &
                                                        (summ_scores_df['shuffle']==shuff)]
                            epoch_arr[m, s, sh, 0] = sub_df['epoch_mean']
                            epoch_arr[m, s, sh, 1] = sub_df['epoch_sem_qu3']
                            epoch_arr[m, s, sh, 2] = sub_df['epoch_qu98']

                            test_arr[m, s, sh, 0] = sub_df['dff_test_acc_mean']
                            test_arr[m, s, sh, 1] = sub_df['dff_test_acc_sem_qu3']
                            test_arr[m, s, sh, 2] = sub_df['dff_test_acc_qu98']
                
                            n_rois[m, s] = sub_df['n_rois']
                            n_runs[m, s, sh] = sub_df['n_runs']

                plot_my_data(ax_list[a][0][c/2][c%2], epoch_arr, celltype, 'epochs', mice, list_sess, n_rois, analys, n_runs)
                plot_my_data(ax_list[a][1][c/2][c%2], test_arr, celltype, 'test_acc', mice, list_sess, n_rois, analys, n_runs)
        
        fig_ep_ab.savefig(os.path.join(scores_dir, 'epochs_AvB.svg'), bbox_inches='tight')
        fig_ep_ac.savefig(os.path.join(scores_dir, 'epochs_AvC.svg'), bbox_inches='tight')
        fig_ep_surp.savefig(os.path.join(scores_dir, 'epochs_surp.svg'), bbox_inches='tight')
        fig_test_ab.savefig(os.path.join(scores_dir, 'test_acc_AvB.svg'), bbox_inches='tight')
        fig_test_ac.savefig(os.path.join(scores_dir, 'test_acc_AvC.svg'), bbox_inches='tight')
        fig_test_surp.savefig(os.path.join(scores_dir, 'test_acc_surp.svg'), bbox_inches='tight')

        plt.close('all')

    elif args.task == 'rem_extra':
        
        model_dir = os.path.join(args.output, 'regr_models')
        dirs = os.listdir(model_dir)
        dirs = [name for name in dirs if os.path.isdir(os.path.join(model_dir, name))]

        for i, direc in enumerate(dirs):
            if direc[0:2] != 'm{}'.format(args.mouse_n):
                continue
            print(direc)
            models = glob.glob(os.path.join(model_dir, direc, 'ep*.pth'))
            jsons = glob.glob(os.path.join(model_dir, direc, 'ep*.json'))
            if len(models) > 0:
                max_ep = max([int(re.findall(r'\d+', os.path.split(mod)[-1])[0]) for mod in models])
            else:
                max_ep = None
                print('    Warning: No models were recorded.')
            for mod in models:
                if mod != os.path.join(model_dir, direc, 'ep{}.pth'.format(max_ep)):
                    os.remove(mod)
            for jso in jsons:
                if jso != os.path.join(model_dir, direc, 'ep{}.json'.format(max_ep)):
                    os.remove(jso)
