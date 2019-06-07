''' 
Nahuel .- Experiment that trains and saves the model in order 
to obtain the generator and discriminator.
'''

import numpy as np
import tensorflow as tf
import pdb
import random
import json
from scipy.stats import mode

import data_utils
import plotting
import model
import utils

from time import time
from math import floor
from mmd import rbf_mmd2, median_pairwise_distance, mix_rbf_mmd2_and_ratio

tf.logging.set_verbosity(tf.logging.ERROR)

# Parameters that will be used
settings = {
"settings_file": "",
"data": "markov",
"num_samples": 14000,
"seq_length": 73,
"num_signals": 1,
"normalise": True,
"cond_dim": 0,
"max_val": 1,
"one_hot": False,
"predict_labels": False,
"scale": 0.1,
"freq_low": 1.0,
"freq_high": 5.0,
"amplitude_low": 0.1,
"amplitude_high": 0.9,
"multivariate_mnist": False,
"full_mnist": False,
"data_load_from": "",
"resample_rate_in_min": 15,
"hidden_units_g": 100,
"hidden_units_d": 100,
"kappa": 1,
"latent_dim": 5,
"batch_mean": False,
"learn_scale": False,
"learning_rate": 0.1,
"batch_size": 28,
"num_epochs": 250,
"D_rounds": 1,
"G_rounds": 4,
"use_time": False,
"WGAN": False,
"WGAN_clip": False,
"shuffle": True,
"wrong_labels": False,
"identifier": "test",
"dp": False,
"l2norm_bound": 1e-05,
"batches_per_lot": 1,
"dp_sigma": 1e-05,
"num_generated_features": 1
}

# Data is loaded (it is already generated)
data_path = './experiments/data/markov_data_dias.npy'
samples = np.load(data_path).item()
samples = samples['samples']
#samples = samples[0]

# Add settings to the local environment 
locals().update(settings)

# Build Model 
Z, X, CG, CD, CS = model.create_placeholders(batch_size, seq_length, latent_dim, 
                                    num_signals, cond_dim)
discriminator_vars = ['hidden_units_d', 'seq_length', 'cond_dim', 'batch_size', 'batch_mean']
discriminator_settings = dict((k, settings[k]) for k in discriminator_vars)
generator_vars = ['hidden_units_g', 'seq_length', 'batch_size', 
                  'num_generated_features', 'cond_dim', 'learn_scale']
generator_settings = dict((k, settings[k]) for k in generator_vars)

CGAN = (cond_dim > 0)
D_loss, G_loss = model.GAN_loss(Z, X, generator_settings, discriminator_settings, 
        kappa, CGAN, CG, CD, CS, wrong_labels=wrong_labels)
D_solver, G_solver, priv_accountant = model.GAN_solvers(D_loss, G_loss, learning_rate, batch_size, 
        total_examples=samples['train'].shape[0], l2norm_bound=l2norm_bound,
        batches_per_lot=batches_per_lot, sigma=dp_sigma, dp=dp)
G_sample = model.generator(Z, **generator_settings, reuse=True, c=CG)

# Evaluacion
# frequency to do visualisations
vis_freq = max(14000//num_samples, 1)
eval_freq = max(7000//num_samples, 1)
# get heuristic bandwidth for mmd kernel from evaluation samples
heuristic_sigma_training = median_pairwise_distance(samples['vali'])
best_mmd2_so_far = 1000
# optimise sigma using that (that's t-hat)
batch_multiplier = 5000//batch_size
eval_size = batch_multiplier*batch_size
eval_eval_size = int(0.2*eval_size)
eval_real_PH = tf.placeholder(tf.float32, [eval_eval_size, seq_length, num_generated_features])
eval_sample_PH = tf.placeholder(tf.float32, [eval_eval_size, seq_length, num_generated_features])
n_sigmas = 2
sigma = tf.get_variable(name='sigma', shape=n_sigmas, initializer=tf.constant_initializer(value=np.power(heuristic_sigma_training, np.linspace(-1, 3, num=n_sigmas))))
mmd2, that = mix_rbf_mmd2_and_ratio(eval_real_PH, eval_sample_PH, sigma)

with tf.variable_scope("SIGMA_optimizer"):
    sigma_solver = tf.train.RMSPropOptimizer(learning_rate=0.05).minimize(-that, var_list=[sigma])
    #sigma_solver = tf.train.AdamOptimizer().minimize(-that, var_list=[sigma])
    #sigma_solver = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(-that, var_list=[sigma])
sigma_opt_iter = 2000
sigma_opt_thresh = 0.001
sigma_opt_vars = [var for var in tf.global_variables() if 'SIGMA_optimizer' in var.name]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Sample 
# Noise from which sample is generated (vis_sample)
vis_Z = model.sample_Z(batch_size, seq_length, latent_dim, use_time)
vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z})
vis_C = None

# Real sample 
vis_real_indices = np.random.choice(len(samples['vali']), size=6)
vis_real = np.float32(samples['vali'][vis_real_indices, :, :])

# Real samples are saved (training set sequences) 
plotting.save_plot_sample(vis_real, 0, identifier + '_real', n_samples=6, 
                            num_epochs=num_epochs)

# Trace is beeing saved with the experiment results
trace = open('./experiments/traces/' + identifier + '.trace.txt', 'w')
trace.write('epoch time D_loss G_loss mmd2 that\n')

# --- train --- #
train_vars = ['batch_size', 'D_rounds', 'G_rounds', 'use_time', 'seq_length', 
              'latent_dim', 'num_generated_features', 'cond_dim', 'max_val', 
              'WGAN_clip', 'one_hot']
train_settings = dict((k, settings[k]) for k in train_vars)

t0 = time()
best_epoch = 0
print('epoch\ttime\tD_loss\tG_loss\tmmd2\tthat')
labels = dict()
labels['train'], labels['vali'], labels['test'] = None, None, None

# Training and generation 
for epoch in range(num_epochs):
    D_loss_curr, G_loss_curr = model.train_epoch(epoch, samples['train'], labels['train'],
                                        sess, Z, X, CG, CD, CS,
                                        D_loss, G_loss,
                                        D_solver, G_solver, 
                                        **train_settings)
    # Generated sequences are visualized 
    vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z})
    plotting.visualise_at_epoch(vis_sample, data, 
            predict_labels, one_hot, epoch, identifier, num_epochs,
            resample_rate_in_min, multivariate_mnist, seq_length, labels=vis_C)

    # mmd2 is computed  
    ## how many samples to evaluate with?
    eval_Z = model.sample_Z(eval_size, seq_length, latent_dim, use_time)
    eval_C = model.sample_C(eval_size, cond_dim, max_val, one_hot)
    eval_sample = np.empty(shape=(eval_size, seq_length, num_signals))
    for i in range(batch_multiplier):
        eval_sample[i*batch_size:(i+1)*batch_size, :, :] = sess.run(G_sample, feed_dict={Z: eval_Z[i*batch_size:(i+1)*batch_size]})
    eval_sample = np.float32(eval_sample)
    eval_real = np.float32(samples['vali'][np.random.choice(len(samples['vali']), size=batch_multiplier*batch_size), :, :])

    eval_eval_real = eval_real[:eval_eval_size]
    eval_test_real = eval_real[eval_eval_size:]
    eval_eval_sample = eval_sample[:eval_eval_size]
    eval_test_sample = eval_sample[eval_eval_size:]

    ## MMD
    # reset ADAM variables
    sess.run(tf.initialize_variables(sigma_opt_vars))
    sigma_iter = 0
    that_change = sigma_opt_thresh*2
    old_that = 0
    while that_change > sigma_opt_thresh and sigma_iter < sigma_opt_iter:
        new_sigma, that_np, _ = sess.run([sigma, that, sigma_solver], feed_dict={eval_real_PH: eval_eval_real, eval_sample_PH: eval_eval_sample})
        that_change = np.abs(that_np - old_that)
        old_that = that_np
        sigma_iter += 1
    opt_sigma = sess.run(sigma)
    mmd2, that_np = sess.run(mix_rbf_mmd2_and_ratio(eval_test_real, eval_test_sample,biased=False, sigmas=sigma))
    
    ## print
    t = time() - t0
    try:
        print('%d\t%.2f\t%.4f\t%.4f\t%.5f\t%.0f' % (epoch, t, D_loss_curr, G_loss_curr, mmd2, that_np))
    except TypeError:       # pdf are missing (format as strings)
        print('%d\t%.2f\t%.4f\t%.4f\t%.5f\t%.0f' % (epoch, t, D_loss_curr, G_loss_curr, mmd2, that_np))
        
    ## save trace
    trace.write(' '.join(map(str, [epoch, t, D_loss_curr, G_loss_curr, mmd2, that_np])) + '\n')
    if epoch % 10 == 0: 
        trace.flush()
        plotting.plot_trace(identifier, xmax=num_epochs, dp=dp)
        
    if shuffle:     # shuffle the training data 
        perm = np.random.permutation(samples['train'].shape[0])
        samples['train'] = samples['train'][perm]
    if labels['train'] is not None:
        labels['train'] = labels['train'][perm]

    if epoch % 50 == 0:
        model.dump_parameters(identifier + '_' + str(epoch), sess)

trace.flush()
plotting.plot_trace(identifier, xmax=num_epochs, dp=dp)
model.dump_parameters(identifier + '_' + str(epoch), sess)
