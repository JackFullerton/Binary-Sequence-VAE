# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 23:09:43 2021

@author: Jack Fullerton
"""
import keras
from keras.layers import Input, Flatten, Dense, Lambda, Reshape
from keras.models import Model
from keras import backend as K
import numpy as np

# DATA STUFF

# Create binary sequence training data (60000 sequences of 1x800)
x_train = np.random.randint(2,size=(60000,1,100))


seq_length = x_train.shape[2]
seq_width = x_train.shape[1]

# Reshape to correct dimensions
x_train = x_train.reshape(x_train.shape[0],seq_width,seq_length,1)
input_shape = (seq_width,seq_length,1)


# MODEL STUFF

# Encoder

# How many latent variables do we want
latent_dim = 20 

input_seq = Input(shape=input_shape, name='encoder_input')
x = Dense(64, activation='relu')(input_seq)
x = Dense(32, activation='relu')(x)
temp_shape = K.int_shape(x)

x = Flatten()(x)
x = Dense(32, activation='relu')(x)

# mean and variance of latent variables
z_mu = Dense(latent_dim, name='latent_mu')(x)   
z_sigma = Dense(latent_dim, name='latent_sigma')(x)  


# Reparameterization trick (Refer to relevant research regarding why this is required)
def sample_z(args):
  z_mu, z_sigma = args
  eps = K.random_normal(shape=(K.shape(z_mu)[0], K.int_shape(z_mu)[1]))
  return z_mu + K.exp(z_sigma / 2) * eps

# Sample vector from latent distribution
z = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([z_mu, z_sigma])

# Print summary of encoder model for testing.
encoder = Model(input_seq, [z_mu, z_sigma, z], name='encoder')
print(encoder.summary())


# Decoder
# These layers and sizes are completely arbitrary and could be experimented with
decoder_input = Input(shape=(latent_dim, ), name='decoder_input')
x = Dense(temp_shape[1]*temp_shape[2]*temp_shape[3], activation='relu')(decoder_input)
x = Reshape((temp_shape[1], temp_shape[2], temp_shape[3]))(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(1, activation='sigmoid', name='decoder_output')(x)

decoder = Model(decoder_input, x, name='decoder')
decoder.summary()

# apply the decoder to sample
z_decoded = decoder(z)

#Define custom loss
#VAE is trained using the sum of reconstruction loss and KL divergence
class CustomLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        
        # Reconstruction loss using binary crossentropy
        recon_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        
        # KL divergence
        kl_loss = -5e-4 * K.mean(1 + z_sigma - K.square(z_mu) - K.exp(z_sigma), axis=-1)
        return recon_loss + kl_loss

    # add custom loss to the class
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

# apply the custom loss to the input sequences and the decoded latent distribution sample
y = CustomLayer()([input_seq, z_decoded])
# y is basically the original sequence after encoding input sequence to mu, sigma, z
# and decoding sampled z values.
#This will be used as output for vae


# VAE
vae = Model(input_seq, y, name='vae')

# Compile with adam optimization
vae.compile(optimizer='adam', loss=None)
vae.summary()

# Train autoencoder
vae.fit(x_train, None, epochs = 10, batch_size = 25)


# Results

# Reformat one sequence, pass through encoder, decoder then format back to int binary array

test_seq = x_train[42][:,:,0]
print( x_train[42][:,:,0])

test_seq = test_seq.reshape(1,seq_width,seq_length,1);
test_mu,test_sigma,test_z = encoder.predict(test_seq)

print(test_mu)
print(test_sigma)
print(test_z)

test_decoded = decoder.predict(test_z)
test_result = test_decoded[0][:,:,0]
test_result = np.round(test_result,2)
test_result = test_result.astype('int32')
print(test_result)

test_mistakes = x_train[42][:,:,0] - test_result
print(np.sum(test_mistakes))
