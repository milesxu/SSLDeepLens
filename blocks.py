"""
resnet identity block
"""

import tensorflow as tf


class ResnetIdentityBlock(tf.keras.Model):
    """
    resnet block
    """

    def __init__(self, filters, downsampling=False, activefun='elu',
                 preactivated=False):
        super(ResnetIdentityBlock, self).__init__(name='')
        self.filters = filters
        self.downsampling = downsampling
        stride = 2 if downsampling else 1

        self.preactivated = preactivated

        self.actin = tf.keras.layers.Activation(activation=activefun)
        self.bnin = tf.keras.layers.BatchNormalization()

        self.conv2a = tf.keras.layers.Conv2D(
            self.filters[0], 1, strides=stride, padding='same',
            activation=activefun,
            kernel_initializer='he_normal')
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(
            self.filters[0], 3, strides=1, padding='same',
            activation=activefun,
            kernel_initializer='he_normal')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(
            self.filters[1], 1, strides=1, padding='same',
            kernel_initializer='he_normal')
        self.bn2c = tf.keras.layers.BatchNormalization()

        self.convshort = tf.keras.layers.Conv2D(
            self.filters[1], 1, strides=stride, padding='same',
            kernel_initializer='he_normal')

    def call(self, inputs, training=False):
        if not self.preactivated:
            prein = self.bnin(inputs)
            prein = self.actin(prein)
        else:
            prein = inputs

        x = self.conv2a(prein)
        x = self.bn2a(x, training=training)
        #x = activefun(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        # channels last default
        increase_dim = inputs.shape[3] != self.filters[1]
        if self.downsampling or increase_dim:
            shortcut = self.convshort(prein)
            x += shortcut
        else:
            x += prein

        return tf.nn.relu(x)


class DownsamplingGroup(tf.keras.Model):
    """
    three resnet with same filter sizes.
    """

    def __init__(self, filters):
        super(DownsamplingGroup, self).__init__(name='')
        self.block1 = ResnetIdentityBlock(filters, downsampling=True)
        self.block2 = ResnetIdentityBlock(filters)
        self.block3 = ResnetIdentityBlock(filters)

    def call(self, inputs, training=False):
        x = self.block1(inputs, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)

        return x


class PreactivatedGroup(tf.keras.Model):
    """
    three resnet with same filter sizes.
    """

    def __init__(self, filters):
        super(PreactivatedGroup, self).__init__(name='')
        self.block1 = ResnetIdentityBlock(filters, preactivated=True)
        self.block2 = ResnetIdentityBlock(filters)
        self.block3 = ResnetIdentityBlock(filters)

    def call(self, inputs, training=False):
        x = self.block1(inputs, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)

        return x
