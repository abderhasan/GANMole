# GANMole
This code uses a Generative Adversarial Network (GAN) to generate melanoma images using Keras.

To run the code:

`$ python generative_adversarial_network`

I used 374 images to train the GAN, and have resized the images to 32x32 due to the resource exhaustion I get faced with on the GPU I'm using when going with higher resolution images. You can download the image from, <a href="https://drive.google.com/drive/folders/14r8ivbgGk4wEH8JXESXS30ONOz0oAZC4?usp=sharing"><strong>here</strong></a>. I would like to mention that I have renamed the images into sequenced number images, that is 0.jpg, 1.jpg, 2.jpg, ..., 373.jpg. You can use <a href="https://github.com/abderhasan/rename_image_files_to_sequence_number_name"><strong>this script</strong></a> to rename the image file names to sequence number names.

Of course, the goal here is not to reach an optimum solution, but rather demonstrate how a GAN can be used to generate melanoma images.

Have fun!
