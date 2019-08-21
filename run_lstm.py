import os
import argparse
import glob
import multiprocessing

from matplotlib import pyplot as plt
import torch
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from sess_util import sess_data_util, sess_plot_util, sess_gen_util, \
                      sess_str_util
from util import data_util, file_util, gen_util, logreg_util, math_util, \
                 plot_util



class PredLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, 
                 dropout=0):
        super(PredLSTM, self).__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(self.input_dim, self.hidden_dim, 
                                  self.num_layers, dropout=dropout)
        self.hidden2pred = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, par_vals, pred_len='half'):

        # par_vals should be seq x batch x feat
        lstm_out, self.hidden = self.lstm(par_vals)
        # lstm_out is: sequence x batch x hidden (one direction)
        if pred_len == 'half':
            pred_len = int(lstm_out.shape[0]/2.)
        lstm_pred = lstm_out[(lstm_out.shape[0] - pred_len):]
        # pass sequences as batches to hidden
        item_space = self.hidden2pred(lstm_pred.view(-1, lstm_pred.shape[-1]))

        return item_space


class ConvPredROILSTM(PredLSTM):
    def __init__(self, hidden_dim, output_dim, num_layers=3, dropout=0, 
                 out_ch=3, gab_pars=4, n_gab=30, run=True):

        PredLSTM.__init__(self, out_ch * 2 + run, hidden_dim, output_dim, 
                          num_layers=num_layers, dropout=dropout)

        self.gab_pars = gab_pars
        self.n_gab  = n_gab
        self.run = run
        self.conv_par = torch.nn.Conv2d(1, out_ch, (1, self.gab_pars))
        self.conv_gab = torch.nn.Conv2d(out_ch, out_ch * 2, (self.n_gab, 1))


    def forward(self, par_vals, pred_len='half'):

        # par_vals should be seq x batch x feat
        seq_len, b_len, _ = par_vals.shape
        gab_data = par_vals[:, :, : -self.run].view(seq_len, b_len, 
                                        self.n_gab, self.gab_pars).contiguous()
        conved_pars = self.conv_par(gab_data.view(-1, 1, self.n_gab, 
                                                  self.gab_pars))
        conved_gabs = self.conv_gab(conved_pars)
        par_vals    = conved_gabs.view(seq_len, b_len, -1)
        # conved_gabs = self.conv_gab(conved_pars.view(-1, ))
        if self.run:
            run_data = par_vals[:, :, -self.run:]
            par_vals = torch.cat((par_vals, run_data), -1)

        roi_space = super(ConvPredROILSTM, self).forward(par_vals, pred_len)

        return roi_space


def run_lstm(mod, dl, device, train=True):
    loss_tot = 0
    for _, (data, targ) in enumerate(dl, 0):
        if train:
            mod.opt.zero_grad()
        # batch by sequence x frames x par -> fr x batch x par
        batch_len, seq_len, n_items = targ.shape
        pred_tr = mod(data.transpose(1, 0).to(device))
        pred_tr = pred_tr.view([seq_len, batch_len, n_items]).transpose(1, 0)
        loss = mod.loss_fn(pred_tr, targ.to(device))
        if train:
            loss.backward()
            mod.opt.step()
        loss_tot += loss.item()

    return loss_tot


def run_dl(mod, dl, device='cpu', train=True):
    if train:
        mod.train()
    else:
        mod.eval()
    if train:
        loss = run_lstm(mod, dl, device, train)
    else:
        with torch.no_grad():
            loss = run_lstm(mod, dl, device, train)
    return loss


def run_sess_lstm(args, sessid):

    if args.parallel and args.plt_bkend is not None:
        plt.switch_backend(args.plt_bkend) # needs to be repeated within joblib

    args.seed = gen_util.seed_all(args.seed, args.device)

    train_p = 0.8
    lr = 1. * 10**(-args.lr_ex)
    if args.conv:
        conv_str = '_conv'
        outch_str = '_{}outch'.format(args.out_ch)
    else:
        conv_str = ''
        outch_str = ''

    # Input output parameters
    n_stim_s  = 0.6
    n_roi_s = 0.3

    # Stim/traces for training
    train_gabfr = 0
    train_post = 0.9 # up to C
    roi_train_pre = 0 # from A
    stim_train_pre   = 0.3 # from preceeding grayscreen

    # Stim/traces for testing (separated for surp vs nonsurp)
    test_gabfr = 3
    test_post  = 0.6 # up to grayscreen
    roi_test_pre = 0 # from D/E
    stim_test_pre   = 0.3 # from preceeding C

    sess = sess_gen_util.init_sessions(sessid, args.datadir, args.mouse_df, 
                                       args.runtype, fulldict=False, 
                                       dend=args.dend)[0]

    analysdir = sess_gen_util.get_analysdir(sess.mouse_n, sess.sess_n, 
                                    sess.layer, stimtype=args.stimtype, 
                                    comp=None)
    dirname = os.path.join(args.output, analysdir)
    file_util.createdir(dirname, print_dir=False)

    # ARG cannot scale ROIs or running BEFOREHAND. Must do after

    # seq x frame x gabor x par
    train_stim_wins, run_stats = sess_data_util.get_stim_data(sess, 
                          args.stimtype, n_stim_s, train_gabfr, stim_train_pre, 
                          train_post, gabk=16)

    xran, train_roi_wins, roi_stats = sess_data_util.get_roi_data(sess, 
                          args.stimtype, n_roi_s, train_gabfr, roi_train_pre, 
                          train_post, gabk=16)

    test_stim_wins = []
    test_roi_wins  = []
    for surp in [0, 1]:
        stim_wins = sess_data_util.get_stim_data(sess, args.stimtype, n_stim_s, 
                           test_gabfr, stim_test_pre, test_post, surp, gabk=16, 
                           run_mean=run_stats[0], run_std=run_stats[1])
        test_stim_wins.append(stim_wins)
        
        roi_wins = sess_data_util.get_roi_data(sess, args.stimtype, n_roi_s,  
                           test_gabfr, roi_test_pre, test_post, surp, gabk=16, 
                           roi_means=roi_stats[0], roi_stds=roi_stats[1])[1]
        test_roi_wins.append(roi_wins)

    n_pars = train_stim_wins.shape[-1] # n parameters (121)
    n_rois = train_roi_wins.shape[-1] # n ROIs

    hyperstr = '{}hd_{}hl_{}lrex_{}bs{}{}'.format(args.hidden_dim, 
                    args.num_layers, args.lr_ex, args.batchsize, outch_str, 
                    conv_str)

    dls = data_util.create_dls(train_stim_wins, train_roi_wins, train_p=train_p, 
                            test_p=0, batchsize=args.batchsize, thresh_cl=0, 
                            train_shuff=True)[0]
    train_dl, val_dl, _ = dls

    test_dls = []
    for s in [0, 1]:
        dl = data_util.init_dl(test_stim_wins[s], test_roi_wins[s], 
                            batchsize=args.batchsize)
        test_dls.append(dl)


    if args.conv:
        lstm = ConvPredROILSTM(args.hidden_dim, n_rois, out_ch=args.out_ch, 
                            num_layers=args.num_layers, dropout=args.dropout)
    else:
        lstm = PredLSTM(n_pars, args.hidden_dim, n_rois, 
                        num_layers=args.num_layers, dropout=args.dropout)

    lstm = lstm.to(args.device)
    lstm.loss_fn = torch.nn.MSELoss(size_average=False)
    lstm.opt = torch.optim.Adam(lstm.parameters(), lr=lr)

    loss_df = pd.DataFrame(np.nan, index=range(args.n_epochs), 
                           columns=['train', 'val'])
    min_val = np.inf
    for ep in range(args.n_epochs):
        print('\n====> Epoch {}'.format(ep))
        if ep == 0:
            train_loss = run_dl(lstm, train_dl, args.device, train=False)    
        else:
            train_loss = run_dl(lstm, train_dl, args.device, train=True)
        val_loss = run_dl(lstm, val_dl, args.device, train=False)
        loss_df['train'].loc[ep] = train_loss/train_dl.dataset.n_samples
        loss_df['val'].loc[ep] = val_loss/val_dl.dataset.n_samples
        print('Training loss  : {}'.format(loss_df['train'].loc[ep]))
        print('Validation loss: {}'.format(loss_df['val'].loc[ep]))

        # record model if training is lower than val, and val reaches a new low
        if ep == 0 or val_loss < min_val:
            prev_model = glob.glob(os.path.join(dirname, '{}_ep*.pth'.format(hyperstr)))
            prev_df = glob.glob(os.path.join(dirname, '{}.csv'.format(hyperstr)))
            min_val = val_loss
            saved_ep = ep
                
            if len(prev_model) == 1 and len(prev_df) == 1:
                os.remove(prev_model[0])
                os.remove(prev_df[0])

            savename = '{}_ep{}'.format(hyperstr, ep)
            savefile = os.path.join('{}'.format(dirname), savename)
        
            torch.save({'net': lstm.state_dict(), 'opt': lstm.opt.state_dict()},
                        '{}.pth'.format(savefile))
        
            file_util.saveinfo(loss_df, hyperstr, dirname, 'csv')


    plot_util.linclab_plt_defaults(font=['Arial', 'Liberation Sans'], 
                                font_dir='../tools/fonts')
    fig, ax = plt.subplots(1)
    for dataset in ['train', 'val']:
        plot_util.plot_traces(ax, range(args.n_epochs), np.asarray(loss_df[dataset]), 
                            label=dataset, title='Average loss (MSE) ({} ROIs)'.format(n_rois))
    fig.savefig(os.path.join(dirname, '{}_loss'.format(hyperstr)))

    savemod = os.path.join(dirname, '{}_ep{}.pth'.format(hyperstr, saved_ep))
    checkpoint = torch.load(savemod)
    lstm.load_state_dict(checkpoint['net']) 

    n_samples = 20
    val_idx = np.random.choice(range(val_dl.dataset.n_samples), n_samples)
    val_samples = val_dl.dataset[val_idx]
    xrans = data_util.get_win_xrans(xran, val_samples[1].shape[1], val_idx.tolist())

    fig, ax = plot_util.init_fig(n_samples, ncols=4, sharex=True, subplot_hei=2, 
                                subplot_wid=5)


    lstm.eval()
    with torch.no_grad():
        batch_len, seq_len, n_items = val_samples[1].shape
        pred_tr = lstm(val_samples[0].transpose(1, 0).to(args.device))
        pred_tr = pred_tr.view([seq_len, batch_len, n_items]).transpose(1, 0)

    for lab, data in zip(['target', 'pred'], [val_samples[1], pred_tr]):
        data = data.numpy()
        for n in range(n_samples):
            roi_n = np.random.choice(range(data.shape[-1]))
            sub_ax = plot_util.get_subax(ax, n)
            plot_util.plot_traces(sub_ax, xrans[n], data[n, :, roi_n], label=lab)
            plot_util.set_ticks(sub_ax, 'x', xran[0], xran[-1], n=7)

    sess_plot_util.plot_labels(ax, train_gabfr, plot_vals='reg', pre=roi_train_pre, 
                            post=train_post)

    fig.suptitle('Target vs predicted validation traces ({} ROIs)'.format(n_rois))
    fig.savefig(os.path.join(dirname, '{}_traces'.format(hyperstr)))




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

        # general parameters
    parser.add_argument('--datadir', default=None, 
                        help=('data directory (if None, uses a directory '
                              'defined below'))
    parser.add_argument('--output', default='lstm_models', help='where to store output')
    parser.add_argument('--plt_bkend', default=None, 
                        help='switch mpl backend when running on server')
    parser.add_argument('--parallel', action='store_true', 
                        help='do sess_n\'s in parallel.')
    parser.add_argument('--seed', default=-1, type=int, help='random seed (-1 for None)')

    parser.add_argument('--dend', default='aibs', help='aibs, extr')
    parser.add_argument('--n_epochs', default=100, type=int)

    parser.add_argument('--conv', action='store_true')
    parser.add_argument('--batchsize', default=40, type=int)
    parser.add_argument('--out_ch', default=10, type=int)
    parser.add_argument('--hidden_dim', default=15, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--lr_ex', default=3, type=int)
    parser.add_argument('--dropout', default=0.2, type=float) 
    
    args = parser.parse_args()

    args.device = 'cpu'

    args.mouse_df = 'mouse_df.csv'
    if args.datadir is None:
        args.datadir = '../data/AIBS'
    args.runtype = 'prod'
    args.layer = 'soma'
    args.stimtype = 'gabors'


    args.omit_sess, args.omit_mice = sess_gen_util.all_omit(args.stimtype, 
                                                            args.runtype)

    
    all_sessids = sess_gen_util.get_sess_vals(args.mouse_df, 'sessid', 
                            runtype=args.runtype, sess_n=[1, 2, 3], 
                            layer=args.layer, min_rois=1, pass_fail='P', 
                            omit_sess=args.omit_sess, omit_mice=args.omit_mice)


    # bsizes =[1, 15, 30] #3
    # outchs = [18, 9, 3]
    # hiddims = [100, 35, 5]
    # numlays = [3, 2, 1]
    # lr_exs = [4, 3, 5]
    # convs = [True, False]
    # args.n_epochs = 0

    if args.parallel:
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(run_sess_lstm)
                (args, sessid) for sessid in all_sessids)
    else:
        for sessid in all_sessids:
            run_sess_lstm(args, sessid)

