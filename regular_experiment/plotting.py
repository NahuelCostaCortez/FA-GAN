''' 
Nahuel .- Plotting samples.
'''

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb
from time import time
from matplotlib.colors import hsv_to_rgb
from pandas import read_table, read_hdf

import paths
from data_utils import scale_data

def visualise_at_epoch(vis_sample, data, predict_labels, one_hot, epoch,
        identifier, num_epochs, resample_rate_in_min, multivariate_mnist,
        seq_length, labels):
    # TODO: what's with all these arguments
    if data == 'mnist':
        if predict_labels:
            n_labels = 1
            if one_hot:
                n_labels = 6
                lab_votes = np.argmax(vis_sample[:, :, -n_labels:], axis=2)
            else:
                lab_votes = vis_sample[:, :, -n_labels:]
            labs, _ = mode(lab_votes, axis=1)
            samps = vis_sample[:, :, :-n_labels]
        else:
            labs = labels
            samps = vis_sample
        if multivariate_mnist:
            save_mnist_plot_sample(samps.reshape(-1, seq_length**2, 1), epoch, identifier, n_samples=6, labels=labs)
        else:
            save_mnist_plot_sample(samps, epoch, identifier, n_samples=6, labels=labs)
    elif 'eICU' in data:
        vis_eICU_patients_downsampled(vis_sample[:6, :, :],
                resample_rate_in_min,
                identifier=identifier,
                idx=epoch)
    else:
        save_plot_sample(vis_sample, epoch, identifier, n_samples=6,
                num_epochs=num_epochs)

    return True

def save_plot_sample(samples, idx, identifier, n_samples=6, num_epochs=None, ncol=2):
    assert n_samples <= samples.shape[0]
    assert n_samples % ncol == 0
    sample_length = samples.shape[1]

    if not num_epochs is None:
        col = hsv_to_rgb((1, 1.0*(idx)/num_epochs, 0.8))
    else:
        col = 'grey'

    x_points = np.arange(sample_length)

    nrow = int(n_samples/ncol)
    fig, axarr = plt.subplots(nrow, ncol, sharex=True, figsize=(6, 6))
    for m in range(nrow):
        for n in range(ncol):
            # first column
            sample = samples[n*nrow + m, :, 0]
            axarr[m, n].plot(x_points, sample, color=col)
            axarr[m, n].set_ylim(-1, 1)
    for n in range(ncol):
        axarr[-1, n].xaxis.set_ticks(range(0, sample_length, int(sample_length/4)))
    fig.suptitle(idx)
    fig.subplots_adjust(hspace = 0.15)
    fig.savefig("./experiments/plots/" + identifier + "_epoch" + str(idx).zfill(4) + ".png")
    plt.clf()
    plt.close()
    return

def plot_sine_evaluation(real_samples, fake_samples, idx, identifier):
    """
    Create histogram of fake (generated) samples frequency, amplitude distribution.
    Also for real samples.
    """
    ### frequency
    seq_length = len(real_samples[0])    # assumes samples are all the same length
    frate = seq_length
    freqs_hz = np.fft.rfftfreq(seq_length)*frate      # this is for labelling the plot
    # TODO, just taking axis 0 for now...
    w_real = np.mean(np.abs(np.fft.rfft(real_samples[:, :, 0])), axis=0)
    w_fake = np.mean(np.abs(np.fft.rfft(fake_samples[:, :, 0])), axis=0)
    ### amplitude
    A_real = np.max(np.abs(real_samples[:, :, 0]), axis=1)
    A_fake = np.max(np.abs(fake_samples[:, :, 0]), axis=1)
    ### now plot
    nrow = 2
    ncol = 2
    fig, axarr = plt.subplots(nrow, ncol, sharex='col', figsize=(6, 6))
    # freq
    axarr[0, 0].vlines(freqs_hz, ymin=np.minimum(np.zeros_like(w_real), w_real), ymax=np.maximum(np.zeros_like(w_real), w_real), color='#30ba50')
    axarr[0, 0].set_title("frequency", fontsize=16)
    axarr[0, 0].set_ylabel("real", fontsize=16)
    axarr[1, 0].vlines(freqs_hz, ymin=np.minimum(np.zeros_like(w_fake), w_fake), ymax=np.maximum(np.zeros_like(w_fake), w_fake), color='#ba4730')
    axarr[1, 0].set_ylabel("generated", fontsize=16)
    # amplitude
    axarr[0, 1].hist(A_real, normed=True, color='#30ba50', bins=30)
    axarr[0, 1].set_title("amplitude", fontsize=16)
    axarr[1, 1].hist(A_fake, normed=True, color='#ba4730', bins=30)

    fig.savefig('./experiments/plots/' + identifier + '_eval' + str(idx).zfill(4) +'.png')
    plt.clf()
    plt.close()
    return True

def plot_trace(identifier, xmax=250, final=False, dp=False):
    """
    """

    trace_path = './experiments/traces/' + identifier + '.trace.txt'
    da = read_table(trace_path, sep=' ')
    nrow = 3
    if dp:
        trace_dp_path = './experiments/traces/' + identifier + '.dptrace.txt'
        da_dp = read_table(trace_dp_path, sep=' ')
        nrow += 1

    ncol=1
    fig, axarr = plt.subplots(nrow, ncol, sharex='col', figsize=(6, 6))

    # D_loss
    d_handle,  = axarr[0].plot(da.epoch, da.D_loss, color='red', label='discriminator')
    axarr[0].set_ylabel('D loss')
#    axarr[0].set_ylim(0.9, 1.6)
    if final:
        #D_ticks = [1.0, 1.2, 1.5]
        D_ticks = [0.5, 1.0, 1.5]
        axarr[0].get_yaxis().set_ticks(D_ticks)
        for tick in D_ticks:
            axarr[0].plot((-10, xmax+10), (tick, tick), ls='dotted', lw=0.5, color='black', alpha=0.4, zorder=0)
    # G loss
    ax_G = axarr[0].twinx()
    g_handle,  = ax_G.plot(da.epoch, da.G_loss, color='green', ls='dashed', label='generator')
    ax_G.set_ylabel('G loss')
    if final:
        G_ticks = [2.5, 5]
        ax_G.get_yaxis().set_ticks(G_ticks)
#        for tick in G_ticks:
#            axarr[0].plot((-10, xmax+10), (tick, tick), ls='dotted', lw=0.5, color='green', alpha=1.0, zorder=0)

    ax_G.spines["top"].set_visible(False)
    ax_G.spines["bottom"].set_visible(False)
    ax_G.spines["right"].set_visible(False)
    ax_G.spines["left"].set_visible(False)
    ax_G.tick_params(bottom='off', right='off')
    axarr[0].legend(handles=[d_handle, g_handle], labels=['discriminator', 'generator'])

    # mmd
    da_mmd = da.loc[:, ['epoch', 'mmd2']].dropna()
    axarr[1].plot(da_mmd.epoch, da_mmd.mmd2, color='purple')
    axarr[1].set_ylabel('MMD$^2$')
    #axarr[1].set_ylim(0.0, 0.04)

    #ax_that = axarr[1].twinx()
    #ax_that.plot(da.that)
    #ax_that.set_ylabel('$\hat{t}$')
    #ax_that.set_ylim(0, 50)
    if final:
        mmd_ticks = [0.01, 0.02, 0.03]
        axarr[1].get_yaxis().set_ticks(mmd_ticks)
        for tick in mmd_ticks:
            axarr[1].plot((-10, xmax+10), (tick, tick), ls='dotted', lw=0.5, color='black', alpha=0.4, zorder=0)

    # log likelihood
    da_ll = da.loc[:, ['epoch', 'll', 'real_ll']].dropna()
    axarr[2].plot(da_ll.epoch, da_ll.ll, color='orange')
    axarr[2].plot(da_ll.epoch, da_ll.real_ll, color='orange', alpha=0.5)
    axarr[2].set_ylabel('likelihood')
    axarr[2].set_xlabel('epoch')
    axarr[2].set_ylim(-750, 100)
    #axarr[2].set_ylim(-10000000, 500)
    if final:
#        ll_ticks = [-1.0*1e7, -0.5*1e7, 0]
        ll_ticks = [-500 ,-250, 0]
        axarr[2].get_yaxis().set_ticks(ll_ticks)
        for tick in ll_ticks:
            axarr[2].plot((-10, xmax+10), (tick, tick), ls='dotted', lw=0.5, color='black', alpha=0.4, zorder=0)

    if dp:
        assert da_dp.columns[0] == 'epoch'
        epochs = da_dp['epoch']
        eps_values = da_dp.columns[1:]
        for eps_string in eps_values:
            if 'eps' in eps_string:
                eps = eps_string[3:]
            else:
                eps = eps_string
            deltas = da_dp[eps_string]
            axarr[3].plot(epochs, deltas, label=eps)
            axarr[3].set_ylabel('delta')
            axarr[3].set_xlabel('epoch')
        axarr[3].legend()

    # beautify
    for ax in axarr:
        #ax.spines["top"].set_visible(True)
        ax.spines["top"].set_color((0, 0, 0, 0.3))
        #ax.spines["bottom"].set_visible(False)
        ax.spines["bottom"].set_color((0, 0, 0, 0.3))
        #ax.spines["right"].set_visible(False)
        ax.spines["right"].set_color((0, 0, 0, 0.3))
        #ax.spines["left"].set_visible(False)
        ax.spines["left"].set_color((0, 0, 0, 0.3))
        ax.tick_params(bottom='off', left='off')
        # make background grey
    #    ax.set_facecolor((0.96, 0.96, 0.96))
        ymin, ymax = ax.get_ylim()
        for x in np.arange(0, xmax+10, 10):
            ax.plot((x, x), (ymin, ymax), ls='dotted', lw=0.5, color='black', alpha=0.40, zorder=0)
        ax.set_xlim(-5, xmax)
        ax.get_yaxis().set_label_coords(-0.11,0.5)

    # bottom one

    fig.savefig('./experiments/traces/' + identifier + '_trace.png')
    fig.savefig('./experiments/traces/' + identifier + '_trace.pdf')
    plt.clf()
    plt.close()
    return True

# --- to do with the model --- #
def plot_parameters(parameters, identifier):
    """
    visualise the parameters of a GAN
    """
    generator_out = parameters['generator/W_out_G:0']
    generator_weights = parameters['generator/rnn/lstm_cell/weights:0'] # split this into four
    generator_matrices = np.split(generator_weights, 4, 1)
    fig, axarr = plt.subplots(5, 1, sharex=True,
            gridspec_kw = {'height_ratios':[0.2, 1, 1, 1, 1]}, figsize=(3,13))

    axarr[0].matshow(generator_out.T, extent=[0,100,0,100])
    axarr[0].set_title('W_out_G')
    axarr[1].matshow(generator_matrices[0])
    axarr[1].set_title('LSTM weights (1)')
    axarr[2].matshow(generator_matrices[1])
    axarr[2].set_title('LSTM weights (2)')
    axarr[3].matshow(generator_matrices[2])
    axarr[3].set_title('LSTM weights (3)')
    axarr[4].matshow(generator_matrices[3])
    axarr[4].set_title('LSTM weights (4)')
    for a in axarr:
        a.set_xlim(0, 100)
        a.set_ylim(0, 100)
        a.spines["top"].set_visible(False)
        a.spines["bottom"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.spines["left"].set_visible(False)
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
#        a.tick_params(bottom='off', left='off', top='off')
    plt.tight_layout()
    plt.savefig('./experiments/plots/' + identifier + '_weights.png')
    return True