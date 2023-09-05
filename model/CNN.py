import tensorflow as tf
from keras import layers, Model

class convBlock(layers.Layer):
    def __init__(self, output_dim, kernel_size=7, stride=3, max_pool=2):
        super().__init__()
        self.convs = tf.keras.Sequential([
            layers.Conv1D(filters=output_dim, kernel_size=kernel_size, padding='same', strides=stride),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling1D(pool_size=max_pool, strides=max_pool)
        ])
        
    def __call__(self, x):
        x = self.convs(x)
        return x
    

class ecg_CNN(Model):
    def __init__(self, channels=[4, 16, 32], dropout_rate=0.1, hidden_dim=512, num_crits=5, **kwargs):
        super().__init__()
        self.conv_blocks = tf.keras.Sequential()
        for channel in channels:
            self.conv_blocks.add(convBlock(channel))
            self.conv_blocks.add(layers.Dropout(dropout_rate))
        self.head = tf.keras.Sequential([
            layers.Dense(hidden_dim),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(num_crits),
        ])
        
    def call(self, x):
        x = self.conv_blocks(x)
        x = tf.reshape(x, [-1, 23*32])
        x = self.head(x)
        return x
    
if __name__ == "__main__":
    x = tf.random.uniform([512, 5000, 12])
    model = ecg_CNN()
    output = model(x)
    print(output.shape)