import tensorflow as tf


class FeatureNet(tf.keras.Model):

    def __init__(self, num_classes):
        super(FeatureNet, self).__init__()

        self.conv_1 = tf.keras.layers.Conv3D(filters=32, kernel_size=7, strides=(2, 2, 2), padding="same", use_bias=True,
                                             name="conv_1")
        self.conv_2 = tf.keras.layers.Conv3D(filters=32, kernel_size=5, strides=(1, 1, 1), padding="same", use_bias=True,
                                              name="conv_2")
        self.conv_3 = tf.keras.layers.Conv3D(filters=64, kernel_size=4, strides=(1, 1, 1), padding="same", use_bias=True,
                                              name="conv_3")
        self.conv_4 = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=(1, 1, 1), padding="same", use_bias=True,
                                             name="conv_4")

        self.bn_1 = tf.keras.layers.BatchNormalization(name="batch_norm_1")

        self.pooling_1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same",
                                                   name="pooling_1")

        self.flatten = tf.keras.layers.Flatten()
        self.fc_1 = tf.keras.layers.Dense(units=128, use_bias=True, name="fc_1")
        self.fc_2 = tf.keras.layers.Dense(units=num_classes, use_bias=True, name="fc_2")
        self.softmax = tf.keras.layers.Softmax()

    def call(self, input, training=False):

        x = self.conv_1(input)
        x = self.bn_1(x)
        x = tf.nn.relu(x)

        x = self.conv_2(x)
        x = tf.nn.relu(x)

        x = self.conv_3(x)
        x = tf.nn.relu(x)

        x = self.conv_4(x)
        x = tf.nn.relu(x)

        x = self.pooling_1(x)
        x = self.flatten(x)

        x = self.fc_1(x)
        x = tf.nn.relu(x)
        x = self.fc_2(x)
        x = self.softmax(x)

        return x
