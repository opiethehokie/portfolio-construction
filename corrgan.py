from mlfinlab.risk_estimators import RiskEstimators
import os
import time

from datetime import date
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils import shuffle
from scipy.cluster import hierarchy
from statsmodels.stats.correlation_tools import corr_clipped
from tensorflow.keras import layers

# https://mlfinlab.readthedocs.io/en/latest/data_generation/corrgan.html
# https://gmarti.gitlab.io/ml/2019/09/22/tf-mlp-gan-repr-correlation-matrices.html
# https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html

tickers = ['VNQ','VEA','LEMB','VNQI','VWO','SGOL','VTI','GBTC','TYD','FMF','PDBC','SCHP'] # avg permutation from make_dataset
n = len(tickers)

batch_size = 128
epochs = 5000

saved_model_name = 'generator.tf'

def make_generator_model():
  model = tf.keras.Sequential()
  model.add(layers.Dense(3 * 3 * 256, use_bias=False, input_shape=(n,)))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  model.add(layers.Reshape((3, 3, 256)))
  assert model.output_shape == (None, 3, 3, 256)
  model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
  assert model.output_shape == (None, 3, 3, 128)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 6, 6, 64)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
  assert model.output_shape == (None, n, n, 1)
  return model

def make_discriminator_model():
  model = tf.keras.Sequential()
  model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[n, n, 1]))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))
  model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))
  model.add(layers.Flatten())
  model.add(layers.Dense(1))
  return model

generator = make_generator_model()
discriminator = make_discriminator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  return real_loss + fake_loss

def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

def make_dataset(corrs):
  corrs = np.array(corrs).reshape(-1, n, n).astype('float32')
  #for i, corr in enumerate(corrs):
  #  permutation = corr.sum(axis=1).argsort()
  #  prows = corr[permutation, :]
  #  corrs[i] = prows[:, permutation]
  corrs = corrs.reshape(len(corrs), n, n, 1).astype('float32')
  dataset = (tf.data.Dataset.from_tensor_slices(corrs).shuffle(corrs.shape[0]).batch(batch_size))
  return dataset

@tf.function
def train_step(images):
  noise = tf.random.normal([batch_size, n])
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True)
    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)
    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)
  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
  gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset):
  for epoch in range(epochs):
    start = time.time()
    for image_batch in dataset:
      train_step(image_batch)
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
  generator.save(saved_model_name)

def sample():
  trained_generator = tf.keras.models.load_model(saved_model_name, compile=False)
  noise = tf.random.normal([1, n])
  generated_image = trained_generator(noise, training=False)
  a, b = np.triu_indices(n, k=1)
  corr = np.array(generated_image[0,:,:,0])
  np.fill_diagonal(corr, 1)
  corr[b, a] = corr[a, b]
  nearest_corr = corr_clipped(corr)
  np.fill_diagonal(nearest_corr, 1)
  nearest_corr[b, a] = nearest_corr[a, b]
  return nearest_corr


if __name__ == "__main__":

  from lib import get_time_interval_returns, get_filtered_returns, get_volume_bar_returns, robust_covariances

  multi_returns = [
    get_volume_bar_returns(tickers, date.today() + relativedelta(days=-59), date.today()),
    get_time_interval_returns(tickers, date.today() + relativedelta(months=-24), date.today(), return_type='fractional', interval='1d'),
    get_time_interval_returns(tickers, date.today() + relativedelta(months=-36), date.today(), interval='1wk'),
    get_time_interval_returns(tickers, date.today() + relativedelta(months=-60), date.today(), interval='1mo'),
    get_filtered_returns(tickers, date.today() + relativedelta(months=-60), date.today())
  ]

  corrs = []
  for returns in multi_returns:
    window_size = len(returns) - 24
    windows = [window for window in returns.rolling(window_size) if window.shape[0] == window_size]
    for window in windows:
      covs = robust_covariances(window)
      corrs.extend([RiskEstimators.cov_to_corr(cov) for cov in covs])

  dataset = make_dataset(shuffle(corrs))

  train(dataset)

  #a, b = np.triu_indices(n, k=1)

  #empirical_corrs = returns.rolling(corr_window_size).corr(pairwise=True)[n * (corr_window_size-1):]
  #empirical_corrs = empirical_corrs.values.reshape(-1, n, n).astype('float32')
  #empirical_corrs = [corr[a, b] for corr in empirical_corrs]

  #generated_corrs = [sample()[a, b] for _ in range(100)]

  # Distribution of pairwise correlations is significantly shifted to the positive

  #plt.hist(generated_corrs, bins=100, alpha=0.5, density=True, label='DCGAN correlations')
  #plt.hist(empirical_corrs, bins=100, alpha=0.5, density=True, label='empirical correlations')
  #plt.axvline(x=np.mean(generated_corrs), color='b', linestyle='dashed', linewidth=2)
  #plt.axvline(x=np.mean(empirical_corrs), color='r', linestyle='dashed', linewidth=2)
  #plt.legend()
  #plt.show()

  #plt.hist(generated_corrs, bins=100, alpha=0.5, density=True, log=True, label='DCGAN correlations')
  #plt.hist(empirical_corrs, bins=100, alpha=0.5, density=True, log=True, label='empirical correlations')
  #plt.axvline(x=np.mean(generated_corrs), color='b', linestyle='dashed', linewidth=2)
  #plt.axvline(x=np.mean(empirical_corrs), color='r', linestyle='dashed', linewidth=2)
  #plt.legend()
  #plt.show()

  #empirical_corr_mats = returns.rolling(corr_window_size).corr(pairwise=True)[n * (corr_window_size-1):]
  #empirical_corr_mats = empirical_corr_mats.values.reshape(-1, n, n).astype('float32')

  #generated_corr_mats = generated_corrs

  # Eigenvalues follow the Marchenko-Pastur distribution

  #def compute_eigenvals(corrs):
  #  eigenvalues = []
  #  for corr in corrs:
  #    eigenvals, _ = np.linalg.eig(corr)
  #    eigenvalues.append(sorted(eigenvals, reverse=True))
  #  return eigenvalues

  #sample_mean_empirical_eigenvals = np.mean(compute_eigenvals(empirical_corr_mats), axis=0)
  #sample_mean_dcgan_eigenvals = np.mean(compute_eigenvals(generated_corr_mats), axis=0)

  #plt.figure(figsize=(10, 6))
  #plt.hist(sample_mean_dcgan_eigenvals, bins=n, density=True, alpha=0.5, label='DCGAN')
  #plt.hist(sample_mean_empirical_eigenvals, bins=n, density=True, alpha=0.5, label='empirical')
  #plt.legend()
  #plt.show()

  # Perron-Frobenius property (first eigenvector has positive entries)

  #def compute_pf_vec(corrs):
  #  pf_vectors = []
  #  for corr in corrs:
  #    eigenvals, eigenvecs = np.linalg.eig(corr)
  #    pf_vector = eigenvecs[:, np.argmax(eigenvals)]
  #    if len(pf_vector[pf_vector < 0]) > len(pf_vector[pf_vector > 0]):
  #      pf_vector = -pf_vector
  #    pf_vectors.append(pf_vector)
  #  return pf_vectors

  #mean_empirical_pf = np.mean(compute_pf_vec(empirical_corr_mats), axis=0)
  #mean_dcgan_pf = np.mean(compute_pf_vec(generated_corr_mats), axis=0)

  #plt.figure(figsize=(10, 6))
  #plt.hist(mean_dcgan_pf, bins=n, density=True, alpha=0.5, label='DCGAN')
  #plt.hist(mean_empirical_pf, bins=n, density=True, alpha=0.5, label='empirical')
  #plt.axvline(x=0, color='k', linestyle='dashed', linewidth=2)
  #plt.axvline(x=np.mean(mean_dcgan_pf), color='b', linestyle='dashed', linewidth=2)
  #plt.axvline(x=np.mean(mean_empirical_pf), color='r', linestyle='dashed', linewidth=2)
  #plt.legend()
  #plt.show()

  # Hierarchical structure of correlations

  #for idx, corr in enumerate(empirical_corr_mats):
  #  dist = 1 - corr
  #  linkage_mat = hierarchy.linkage(dist[a, b], method="ward")
  #  permutation = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(linkage_mat, dist[a, b]))
  #  prows = corr[permutation, :]
  #  ordered_corr = prows[:, permutation]
  #  plt.pcolormesh(ordered_corr)
  #  plt.colorbar()
  #  plt.show()
  #  if idx > 5:
  #    break

  #for idx, corr in enumerate(generated_corr_mats):
  #  dist = 1 - corr
  #  linkage_mat = hierarchy.linkage(dist[a, b], method="ward")
  #  permutation = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(linkage_mat, dist[a, b]))
  #  prows = corr[permutation, :]
  #  ordered_corr = prows[:, permutation]
  #  plt.pcolormesh(corr)
  #  plt.colorbar()
  #  plt.show()
  #  if idx > 5:
  #    break

  # Scale free property of MST

  #def compute_degree_counts(correls):
  #  all_counts = []
  #  for corr in correls:
  #    dist = (1 - corr) / 2
  #    G = nx.from_numpy_matrix(dist) 
  #    mst = nx.minimum_spanning_tree(G)
  #    degrees = {i: 0 for i in range(len(corr))}
  #    for edge in mst.edges:
  #      degrees[edge[0]] += 1
  #      degrees[edge[1]] += 1
  #    degrees = pd.Series(degrees).sort_values(ascending=False)
  #    cur_counts = degrees.value_counts()
  #    counts = np.zeros(len(corr))
  #    for i in range(len(corr)):
  #      if i in cur_counts:
  #        counts[i] = cur_counts[i] 
  #    all_counts.append(counts / (len(corr) - 1))
  #  return all_counts

  #mean_dcgan_counts = np.mean(compute_degree_counts(generated_corr_mats), axis=0)
  #mean_dcgan_counts = (pd.Series(mean_dcgan_counts).replace(0, np.nan))

  #mean_empirical_counts = np.mean(compute_degree_counts(empirical_corr_mats), axis=0)
  #mean_empirical_counts = (pd.Series(mean_empirical_counts).replace(0, np.nan))

  #plt.figure(figsize=(10, 6))
  #plt.scatter(mean_dcgan_counts.index, mean_dcgan_counts, label='DCGAN')
  #plt.scatter(mean_empirical_counts.index, mean_empirical_counts, label='empirical')
  #plt.legend()
  #plt.show()

  #plt.figure(figsize=(10, 6))
  #plt.scatter(np.log10(mean_dcgan_counts.index), np.log10(mean_dcgan_counts), label='DCGAN')
  #plt.scatter(np.log10(mean_empirical_counts.index), np.log10(mean_empirical_counts), label='empirical')
  #plt.legend()
  #plt.show()
