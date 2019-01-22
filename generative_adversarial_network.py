'''
This code enables you to generate melanoma images using a Generative Adversarial Network (GAN)
@author: Abder-Rahman Ali
abder@cs.stir.ac.uk
'''

import keras
from keras import layers
import numpy as np
import cv2
import os
from keras.preprocessing import image

latent_dimension = 32
height = 32
width = 32
channels = 3
iterations = 10000
number_of_images = 374
real_images = []

# paths to the training and results directories
train_directory = '//train'
results_directory = '/results'

# GAN generator
generator_input = keras.Input(shape=(latent_dimension,))

# transform the input into a 16x16 128-channel feature map
x = layers.Dense(128*16*16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16,16,128))(x)

x = layers.Conv2D(256,5,padding='same')(x)
x = layers.LeakyReLU()(x)

# upsample to 32x32
x = layers.Conv2DTranspose(256,4,strides=2,padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256,5,padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256,5,padding='same')(x)
x = layers.LeakyReLU()(x)

# a 32x32 1-channel feature map is generated (i.e. shape of image)
x = layers.Conv2D(channels,7,activation='tanh',padding='same')(x)
# instantiae the generator model, which maps the input of shape (latent dimension) into an image of shape (32,32,1)
generator = keras.models.Model(generator_input,x)
generator.summary()

# GAN discriminator
discriminator_input = layers.Input(shape=(height,width,channels))

x = layers.Conv2D(128,3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128,4,strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128,4,strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128,4,strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

# dropout layer
x = layers.Dropout(0.4)(x)

# classification layer
x = layers.Dense(1,activation='sigmoid')(x)

# instantiate the discriminator model, and turn a (32,32,1) input
# into a binary classification decision (fake or real)
discriminator = keras.models.Model(discriminator_input,x)
discriminator.summary()

discriminator_optimizer = keras.optimizers.RMSprop(
	lr=0.0008,
	clipvalue=1.0,
	decay=1e-8)

discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

# adversarial network
discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dimension,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input,gan_output)

gan_optimizer = keras.optimizers.RMSprop(
	lr=0.0004,
	clipvalue=1.0,
	decay=1e-8)

gan.compile(optimizer=gan_optimizer,loss='binary_crossentropy')

for step in range(iterations):
	# sample random points in the latent space
	random_latent_vectors = np.random.normal(size=(number_of_images,latent_dimension))
	# decode the random latent vectors into fake images
	generated_images = generator.predict(random_latent_vectors)

	for root, dirs, files in os.walk(train_directory):
		for i in range(number_of_images):
			img = cv2.imread(root + '/' + str(i) + '.jpg')
			real_images.append(img)
	
	# combine fake images with real images
	combined_images = np.concatenate([generated_images,real_images])
	# assemble labels and discrminate between real and fake images
	labels = np.concatenate([np.ones((number_of_images,1)),np.zeros((number_of_images,1))])
	# add random noise to the labels
	labels = labels + 0.05 * np.random.random(labels.shape)
	# train the discriminator
	discriminator_loss = discriminator.train_on_batch(combined_images,labels)
	random_latent_vectors = np.random.normal(size=(number_of_images,latent_dimension))
	# assemble labels that classify the images as "real", which is not true
	misleading_targets = np.zeros((number_of_images,1))
	# train the generator via the GAN model, where the discriminator weights are frozen
	adversarial_loss = gan.train_on_batch(random_latent_vectors,misleading_targets)
	gan.save_weights('gan.h5')
	print'discriminator loss: ' 
	print discriminator_loss
	print 'adversarial loss: '
	print adversarial_loss
	
	# show the generated images for each step
	for i in range(generated_images.shape[0]):	
		img = image.array_to_img(generated_images[i] * 255.)
		img.save(os.path.join(results_directory,'generated_melanoma_image' + str(i) + '-' + str(step) + '.jpg'))

	# emoty the real_images list so images are not added up to the next iteration
	real_images = []
