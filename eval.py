# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from preprocessing import preprocessing_factory
import reader
import model
import time
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.io as io
import glob as gb
import cv2
from skimage import data_dir

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                           'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
tf.app.flags.DEFINE_string("model_file", "/home/z/fast-neural-style-tensorflow-master/models/number/fast-style-model.ckpt-done", "")
tf.app.flags.DEFINE_string("image_dir", "img", "")

FLAGS = tf.app.flags.FLAGS
def get_image_paths(folder): #这个函数的作用的获取文件的列表,注释部分是获取
    # return (os.path.join(folder, f)
    #     for f in os.listdir(folder)
    #         if 'png' in f)
    return gb.glob(os.path.join(folder, '*.png'))


def main(_):
    pictures = get_image_paths(FLAGS.image_dir)
    print(pictures)
    for picture in pictures:
        # lena = mpimg.imread(picture)
        # plt.imshow(lena)
        # plt.show()

        with open(picture, 'rb') as img:
            with tf.Session().as_default() as sess:
                if picture.lower().endswith('png'):
                    image = sess.run(tf.image.decode_png(img.read()))
                else:
                    image = sess.run(tf.image.decode_jpeg(img.read()))
                height = image.shape[0]
                width = image.shape[1]
        tf.logging.info('Image size: %dx%d' % (width, height))

        with tf.Graph().as_default():
            with tf.Session().as_default() as sess:

                # Read image data.
                image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                    FLAGS.loss_model,
                    is_training=False)
                image = reader.get_image(picture, height, width, image_preprocessing_fn)

                # Add batch dimension
                image = tf.expand_dims(image, 0)

                generated = model.net(image, training=False)
                generated = tf.cast(generated, tf.uint8)

                # Remove batch dimension
                generated = tf.squeeze(generated, [0])

                # Restore model variables.
                saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                # Use absolute path
                FLAGS.model_file = os.path.abspath(FLAGS.model_file)
                saver.restore(sess, FLAGS.model_file)

                # Make sure 'generated' directory exists.
                if '.png' == picture[-4:]:
                	picture = picture[:-4]
                generated_file = 'generated/'+picture+'.jpg'
                if os.path.exists('generated/img') is False:
                    os.makedirs('generated/img')

                # Generate and write image data to file.
                with open(generated_file, 'wb') as img:
                    start_time = time.time()
                    img.write(sess.run(tf.image.encode_jpeg(generated)))
                    end_time = time.time()
                    tf.logging.info('Elapsed time: %fs' % (end_time - start_time))

                    tf.logging.info('Done. Please check %s.' % generated_file)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
