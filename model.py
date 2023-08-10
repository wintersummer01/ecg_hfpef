import tensorflow as tf
from keras import layers, Model

class residualBlock(layers.Layer):
    def __init__(self, input_dim, output_dim, kernel_size, stride, dropout_rate):
        super().__init__()
        # For skip connection
        self.skip = tf.keras.Sequential([
            layers.MaxPooling1D(pool_size=stride, strides=stride),
            layers.Conv1D(filters=output_dim, kernel_size=1)])
        # For layer1
        self.conv1 = layers.Conv1D(filters=input_dim, kernel_size=kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization([0, 1])
        # For layer2
        self.conv2 = layers.Conv1D(filters=output_dim, kernel_size=kernel_size, padding='same', strides=stride)
        self.bn2 = layers.BatchNormalization([0, 1])
        # For common
        self.activation = layers.ReLU()
        self.dropout = layers.Dropout(dropout_rate)

    def __call__(self, x):
        skip_connection = self.skip(x)
        # layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        # layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x += skip_connection
        x = self.activation(x)
        x = self.dropout(x)
        return x
    

class ecg2Hfpef(Model):
    def __init__(self, dimensions, kernel_sizes, strides, dropout_rate):
        super().__init__()
        self.residual_blocks = tf.keras.Sequential()
        for i, (kernel_size, stride) in enumerate(zip(kernel_sizes, strides)):
            self.residual_blocks.add(
                residualBlock(dimensions[i], dimensions[i+1], kernel_size, stride, dropout_rate)
            )
        self.projection = layers.Dense(1)

    def call(self, x):
        # Resnet blocks
        x = self.residual_blocks(x)

        # Flatten and Linear layer
        x = tf.reshape(x, [1, -1])
        H2fpef_score = self.projection(x)
        return H2fpef_score
