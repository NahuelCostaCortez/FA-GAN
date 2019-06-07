''' 
Nahuel .- Everything that has to with the model.
'''

import tensorflow as tf
import tensorflow as tf
import numpy as np
import data_utils
import pdb
import json
from mod_core_rnn_cell_impl import LSTMCell          #modified to allow initializing bias in lstm
tf.logging.set_verbosity(tf.logging.ERROR)
import mmd

from differential_privacy.dp_sgd.dp_optimizer import dp_optimizer
from differential_privacy.dp_sgd.dp_optimizer import sanitizer
from differential_privacy.privacy_accountant.tf import accountant

# --- to do with latent space --- #

def sample_Z(batch_size, seq_length, latent_dim, use_time=False, use_noisy_time=False):
    sample = np.float32(np.random.normal(size=[batch_size, seq_length, latent_dim]))
    if use_time:
        print('WARNING: use_time has different semantics')
        sample[:, :, 0] = np.linspace(0, 1.0/seq_length, num=seq_length)
    return sample

# --- to do with training --- #

def train_epoch(epoch, samples, labels, sess, Z, X, CG, CD, CS, D_loss, G_loss, D_solver, G_solver, 
                batch_size, use_time, D_rounds, G_rounds, seq_length, 
                latent_dim, num_generated_features, cond_dim, max_val, WGAN_clip, one_hot):
    """
    Train generator and discriminator for one epoch.
    """
    for batch_idx in range(0, int(len(samples) / batch_size) - (D_rounds + (cond_dim > 0)*G_rounds), D_rounds + (cond_dim > 0)*G_rounds):
        # update the discriminator
        for d in range(D_rounds):
            X_mb, Y_mb = data_utils.get_batch(samples, batch_size, batch_idx + d, labels)
            Z_mb = sample_Z(batch_size, seq_length, latent_dim, use_time)
            if cond_dim > 0:
                # CGAN
                Y_mb = Y_mb.reshape(-1, cond_dim)
                if one_hot:
                    # change all of the labels to a different one
                    offsets = np.random.choice(cond_dim-1, batch_size) + 1
                    new_labels = (np.argmax(Y_mb, axis=1) + offsets) % cond_dim
                    Y_wrong = np.zeros_like(Y_mb)
                    Y_wrong[np.arange(batch_size), new_labels] = 1
                else:
                    # flip all of the bits (assuming binary...)
                    Y_wrong = 1 - Y_mb
                _ = sess.run(D_solver, feed_dict={X: X_mb, Z: Z_mb, CD: Y_mb, CS: Y_wrong, CG: Y_mb})
            else:
                _ = sess.run(D_solver, feed_dict={X: X_mb, Z: Z_mb})
            if WGAN_clip:
                # clip the weights
                _ = sess.run([clip_disc_weights])
        # update the generator
        for g in range(G_rounds):
            if cond_dim > 0:
                # note we are essentially throwing these X_mb away...
                X_mb, Y_mb = data_utils.get_batch(samples, batch_size, batch_idx + D_rounds + g, labels)
                _ = sess.run(G_solver,
                        feed_dict={Z: sample_Z(batch_size, seq_length, latent_dim, use_time=use_time), CG: Y_mb})
            else:
                _ = sess.run(G_solver,
                        feed_dict={Z: sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)})
    # at the end, get the loss
    if cond_dim > 0:
        D_loss_curr, G_loss_curr = sess.run([D_loss, G_loss], feed_dict={X: X_mb, Z: sample_Z(batch_size, seq_length, latent_dim, use_time=use_time), CG: Y_mb, CD: Y_mb})
        D_loss_curr = np.mean(D_loss_curr)
        G_loss_curr = np.mean(G_loss_curr)
    else:
        D_loss_curr, G_loss_curr = sess.run([D_loss, G_loss], feed_dict={X: X_mb, Z: sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)})
        D_loss_curr = np.mean(D_loss_curr)
        G_loss_curr = np.mean(G_loss_curr)
    return D_loss_curr, G_loss_curr

# --- loss --- #
def GAN_loss(Z, X, generator_settings, discriminator_settings, kappa, cond, CG, CD, CS, wrong_labels=False):
    if cond:
        # C-GAN
        G_sample = generator(Z, **generator_settings, c=CG)
        D_real, D_logit_real =  discriminator(X, **discriminator_settings, c=CD)
        D_fake, D_logit_fake = discriminator(G_sample, reuse=True, **discriminator_settings, c=CG)
        
        if wrong_labels:
            # the discriminator must distinguish between real data with fake labels and real data with real labels, too
            D_wrong, D_logit_wrong = discriminator(X, reuse=True, **discriminator_settings, c=CS)
    else:
        # normal GAN
        G_sample = generator(Z, **generator_settings)
        D_real, D_logit_real  = discriminator(X, **discriminator_settings)
        D_fake, D_logit_fake = discriminator(G_sample, reuse=True, **discriminator_settings)

    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)), 1)
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)), 1)

    D_loss = D_loss_real + D_loss_fake

    if cond and wrong_labels:
        D_loss = D_loss + D_loss_wrong

    #G_loss = tf.reduce_mean(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)), axis=1))
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)), 1)
    
    return D_loss, G_loss

# -- minimize loss functions -- #
def GAN_solvers(D_loss, G_loss, learning_rate, batch_size, total_examples, 
        l2norm_bound, batches_per_lot, sigma, dp=False):
    """
    Optimizers
    """
    discriminator_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
    generator_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]

	D_loss_mean_over_batch = tf.reduce_mean(D_loss)
	D_solver = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(D_loss_mean_over_batch, var_list=discriminator_vars)
	priv_accountant = None
		
    G_loss_mean_over_batch = tf.reduce_mean(G_loss)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss_mean_over_batch, var_list=generator_vars)
    return D_solver, G_solver, priv_accountant

# --- to do with the model --- #
def create_placeholders(batch_size, seq_length, latent_dim, num_generated_features, cond_dim):
    Z = tf.placeholder(tf.float32, [batch_size, seq_length, latent_dim])
    X = tf.placeholder(tf.float32, [batch_size, seq_length, num_generated_features])
    CG = tf.placeholder(tf.float32, [batch_size, cond_dim])
    CD = tf.placeholder(tf.float32, [batch_size, cond_dim])
    CS = tf.placeholder(tf.float32, [batch_size, cond_dim])
    return Z, X, CG, CD, CS

# --- generator --- #
def generator(z, hidden_units_g, seq_length, batch_size, num_generated_features, reuse=False, parameters=None, cond_dim=0, c=None, learn_scale=True):
    """
    If parameters are supplied, initialise as such
    """
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()
        if parameters is None:
            W_out_G_initializer = tf.truncated_normal_initializer()
            b_out_G_initializer = tf.truncated_normal_initializer()
            scale_out_G_initializer = tf.constant_initializer(value=1.0)
            lstm_initializer = None
            bias_start = 1.0
        else:
            W_out_G_initializer = tf.constant_initializer(value=parameters['generator/W_out_G:0'])
            b_out_G_initializer = tf.constant_initializer(value=parameters['generator/b_out_G:0'])
            try:
                scale_out_G_initializer = tf.constant_initializer(value=parameters['generator/scale_out_G:0'])
            except KeyError:
                scale_out_G_initializer = tf.constant_initializer(value=1)
                assert learn_scale
            lstm_initializer = tf.constant_initializer(value=parameters['generator/rnn/lstm_cell/weights:0'])
            bias_start = parameters['generator/rnn/lstm_cell/biases:0']

        W_out_G = tf.get_variable(name='W_out_G', shape=[hidden_units_g, num_generated_features], initializer=W_out_G_initializer)
        b_out_G = tf.get_variable(name='b_out_G', shape=num_generated_features, initializer=b_out_G_initializer)
        scale_out_G = tf.get_variable(name='scale_out_G', shape=1, initializer=scale_out_G_initializer, trainable=learn_scale)
        if cond_dim > 0:
            # CGAN!
            assert not c is None
            repeated_encoding = tf.stack([c]*seq_length, axis=1)
            inputs = tf.concat([z, repeated_encoding], axis=2)

            #repeated_encoding = tf.tile(c, [1, tf.shape(z)[1]])
            #repeated_encoding = tf.reshape(repeated_encoding, [tf.shape(z)[0], tf.shape(z)[1], cond_dim])
            #inputs = tf.concat([repeated_encoding, z], 2)
        else:
            inputs = z

        cell = LSTMCell(num_units=hidden_units_g,
                           state_is_tuple=True,
                           initializer=lstm_initializer,
                           bias_start=bias_start,
                           reuse=reuse)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            sequence_length=[seq_length]*batch_size,
            inputs=inputs)
        rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, hidden_units_g])
        logits_2d = tf.matmul(rnn_outputs_2d, W_out_G) + b_out_G
        output_2d = tf.nn.tanh(logits_2d)
        output_3d = tf.reshape(output_2d, [-1, seq_length, num_generated_features])
    return output_3d

# --- discriminator --- #
def discriminator(x, hidden_units_d, seq_length, batch_size, reuse=False, 
        cond_dim=0, c=None, batch_mean=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        W_out_D = tf.get_variable(name='W_out_D', shape=[hidden_units_d, 1],
                initializer=tf.truncated_normal_initializer())
        b_out_D = tf.get_variable(name='b_out_D', shape=1,
                initializer=tf.truncated_normal_initializer())

        if cond_dim > 0:
            assert not c is None
            repeated_encoding = tf.stack([c]*seq_length, axis=1)
            inputs = tf.concat([x, repeated_encoding], axis=2)
        else:
            inputs = x
        # add the average of the inputs to the inputs (mode collapse?
        if batch_mean:
            mean_over_batch = tf.stack([tf.reduce_mean(x, axis=0)]*batch_size, axis=0)
            inputs = tf.concat([x, mean_over_batch], axis=2)

        cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units_d, 
                state_is_tuple=True,
                reuse=reuse)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            inputs=inputs)
        logits = tf.einsum('ijk,km', rnn_outputs, W_out_D) + b_out_D
        output = tf.nn.sigmoid(logits)
    return output, logits

# --- to do with saving/loading --- #

def dump_parameters(identifier, sess):
    """
    Save model parmaters to a numpy file
    """
    dump_path = './experiments/parameters/' + identifier + '.npy'
    model_parameters = dict()
    for v in tf.trainable_variables():
        model_parameters[v.name] = sess.run(v)
    np.save(dump_path, model_parameters)
    print('Recorded', len(model_parameters), 'parameters to', dump_path)
    return True

def load_parameters(identifier):
    """
    Load parameters from a numpy file
    """
    load_path = './experiments/parameters/' + identifier + '.npy'
    model_parameters = np.load(load_path).item()
    return model_parameters

# --- to do with trained models --- #

def sample_trained_model(settings, epoch, num_samples, Z_samples=None, C_samples=None):
    """
    Return num_samples samples from a trained model described by settings dict
    """
    # if settings is a string, assume it's an identifier and load
    if type(settings) == str:
        settings = json.load(open('./experiments/settings/' + settings + '.txt', 'r'))
    print('Sampling', num_samples, 'samples from', settings['identifier'], 'at epoch', epoch)
    # get the parameters, get other variables
    parameters = load_parameters(settings['identifier'] + '_' + str(epoch))
    # create placeholder, Z samples
    Z = tf.placeholder(tf.float32, [num_samples, settings['seq_length'], settings['latent_dim']])
    CG = tf.placeholder(tf.float32, [num_samples, settings['cond_dim']])
    if Z_samples is None:
        Z_samples = sample_Z(num_samples, settings['seq_length'], settings['latent_dim'], settings['use_time'], use_noisy_time=False)
    else:
        assert Z_samples.shape[0] == num_samples
    # create the generator (GAN or CGAN)
    if C_samples is None:
        # normal GAN
        G_samples = generator(Z, settings['hidden_units_g'], settings['seq_length'], 
                              num_samples, settings['num_generated_features'], 
                              reuse=False, parameters=parameters, cond_dim=settings['cond_dim'])
    else:
        assert C_samples.shape[0] == num_samples
        # CGAN
        G_samples = generator(Z, settings['hidden_units_g'], settings['seq_length'], 
                              num_samples, settings['num_generated_features'], 
                              reuse=False, parameters=parameters, cond_dim=settings['cond_dim'], c=CG)
    # sample from it 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if C_samples is None:
            real_samples = sess.run(G_samples, feed_dict={Z: Z_samples})
        else:
            real_samples = sess.run(G_samples, feed_dict={Z: Z_samples, CG: C_samples})
    tf.reset_default_graph()
    return real_samples