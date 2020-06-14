# This code is modified from the repository
# https://github.com/bhpfelix/Variational-Autoencoder-PyTorch


import numpy as np

#
# USE_CUDA = True
# try:
#     torch.cuda.current_device()
# except:
#     USE_CUDA = False

import tensorflow as tf
from tensorflow import keras
#
# def create_encoder_model(latent_dim=300):
#     input_layer = tf.keras.layers.Input((64,64,3))
#     x = tf.keras.layers.Conv2D(filters=100,kernel_size=(7,7), strides=(2,2),padding='same')(input_layer)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU(0.2)(x)
#
#     x = tf.keras.layers.Conv2D(filters=150, kernel_size=(4, 4), strides=(2, 2),padding='same')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU(0.2)(x)
#
#     x = tf.keras.layers.Conv2D(filters=250, kernel_size=(4, 4), strides=(2, 2),padding='same')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU(0.2)(x)
#     x = tf.keras.layers.Flatten()(x)
#
#     latent = tf.keras.layers.Dense(latent_dim+latent_dim)(x)
#     # latent_mean = tf.keras.layers.Dense(300)(x)
#     encoder_model = tf.keras.models.Model(inputs=input_layer, outputs=latent)
#     return encoder_model


def create_encoder_model(latent_dim=300):
    input_layer = tf.keras.layers.Input((64,64,3))
    x = tf.keras.layers.Conv2D(filters=100,kernel_size=(7,7), strides=(2,2),padding='same')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(filters=150, kernel_size=(4, 4), strides=(2, 2),padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(filters=250, kernel_size=(4, 4), strides=(2, 2),padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Flatten()(x)

    latent1 = tf.keras.layers.Dense(latent_dim)(x)
    latent2 = tf.keras.layers.Dense(latent_dim)(x)
    latent = tf.keras.layers.concatenate([latent1, latent2])
    # latent_mean = tf.keras.layers.Dense(300)(x)
    encoder_model = tf.keras.models.Model(inputs=input_layer, outputs=latent)
    return encoder_model


#
# def create_decoder_model(latent=300):
#     input_layer = tf.keras.layers.Input((latent))
#     x = tf.keras.layers.Dense(8*8*250)(input_layer)
#     x = tf.keras.layers.ReLU()(x)
#     x = tf.keras.layers.Reshape((8,8,250))(x)
#
#     x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
#     x = tf.keras.layers.Conv2D(filters=150, kernel_size=(3, 3),padding='same')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU(0.2)(x)
#
#     x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
#     x = tf.keras.layers.Conv2D(filters=100, kernel_size=(3, 3),padding='same')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU(0.2)(x)
#
#     x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
#     x = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3),padding='same')(x)
#     # x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Activation(activation='sigmoid')(x)
#
#     model = tf.keras.models.Model(inputs=input_layer,outputs=x)
#     return model
#


def create_decoder_model(latent=300):
    input_layer = tf.keras.layers.Input((latent))
    x = tf.keras.layers.Dense(8*8*250)(input_layer)
    x = tf.keras.layers.Reshape((8,8,250))(x)

    x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
    x = tf.keras.layers.Conv2D(filters=150, kernel_size=(3, 3),padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=100, kernel_size=(3, 3),padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3),padding='same')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=input_layer,outputs=x)
    return model
# class stn_tf()

class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = create_encoder_model(self.latent_dim)

    self.decoder = create_decoder_model(self.latent_dim)

  # @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)

    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits


def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


# def compute_loss_tf(model, x, target):
#   mean, logvar = model.encode(x)
#   z = model.reparameterize(mean, logvar)
#   x_logit = model.decode(z)
#   # tf.keras.losses.binary_crossentropy(target, x_logit)
#   cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=target)
#   logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
#   logpz = log_normal_pdf(z, 0., 0.)
#   logqz_x = log_normal_pdf(z, mean, logvar)
#   return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def compute_loss_tf(x, recon_x, mean, logvar, reduction_type='mean'):
    if reduction_type == 'mean':
        BCE_tf = tf.keras.losses.BinaryCrossentropy()(x, recon_x)
        KLD_tf = (-0.5 * tf.reduce_sum(1 + logvar - (tf.pow(mean, 2) + tf.exp(logvar))))/np.prod(x.shape)
    elif reduction_type == 'sum':
        BCE_tf = tf.keras.losses.BinaryCrossentropy()(x, recon_x)*np.prod(x.shape)
        KLD_tf = -0.5 * tf.reduce_sum(1 + logvar - (tf.pow(mean, 2) + tf.exp(logvar)))
    else:
        raise NotImplementedError("Only reduction of type 'mean' or 'sum' are avaialbe")
    return BCE_tf + KLD_tf

def compute_loss_from_model(target, origin_image, model):
    mean, logvar = model.encode(origin_image)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    loss = compute_loss_tf(x=target,
                           recon_x=x_logit,
                           mean=mean,
                           logvar=logvar)
    return loss, x_logit
    # cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=origin_image)
    # logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    # logpz = log_normal_pdf(z, 0., 0.)
    # logqz_x = log_normal_pdf(z, mean, logvar)
    # return -tf.reduce_mean(logpx_z + logpz - logqz_x)


#
# def compute_loss_tf(x, target, mean, logvar):
#   # mean, logvar = model.encode(x)
#   z = model.reparameterize(mean, logvar)
#   x_logit = model.decode(z)
#   # tf.keras.losses.binary_crossentropy(target, x_logit)
#   cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=target)
#   logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
#   logpz = log_normal_pdf(z, 0., 0.)
#   logqz_x = log_normal_pdf(z, mean, logvar)
#   return -tf.reduce_mean(logpx_z + logpz - logqz_x)
#
#


def get_optimizer():
    # vae_optimizer = tf.keras.optimizers.Adam(1e-4)
    #betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    vae_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8)
    # vae_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0000p1, epsilon=1e-8)
    return vae_optimizer



# @tf.function
def train_step(model, x, target, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
      loss, x_logit = compute_loss_from_model(target=target,origin_image=x, model=model)
      # print(f"tf_loss:{loss}")

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss,x_logit