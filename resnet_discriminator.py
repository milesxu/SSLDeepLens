"""
discriminator with resnet and ssl
"""

import tensorflow as tf
from blocks import DownsamplingGroup, PreactivatedGroup


class ResnetDiscriminator(tf.keras.Model):
    """
    resnet discriminator
    """

    def __init__(self, image_width):
        super(ResnetDiscriminator, self).__init__(name='')

        self.conv2d1 = tf.keras.layers.Conv2D(
            32, 7, padding='same', activation='elu', kernel_initializer='glorot_uniform')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.block1 = PreactivatedGroup((16, 32))
        self.block2 = DownsamplingGroup((32, 64))
        self.block3 = DownsamplingGroup((64, 128))
        self.block4 = DownsamplingGroup((128, 256))
        self.block5 = DownsamplingGroup((256, 512))

        self.pool = tf.keras.layers.AveragePooling2D(
            pool_size=image_width, strides=1)

    def call(self, inputs, training=False):
        x = self.conv2d1(inputs)
        x = self.bn1(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)

        x = self.pool(x)

        return x
