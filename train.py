import argparse
from tqdm import tqdm
import tensorflow as tf
from model import ecg2Hfpef
from utils import makeRandomDataset
from config import KERNEL_SIZE, DIMENSION, STRIDE, DROPOUT_RATE, LEARNING_RATE

def train_model( num_epoch, lr, **train_args):
    model = ecg2Hfpef(**train_args)
    train_set = makeRandomDataset()

    optimizer = tf.optimizers.Adam(learning_rate=lr)
    mse = tf.losses.mean_squared_error

    for i in tqdm(range(num_epoch)):
        for ecg, score in train_set:
            # Forward
            with tf.GradientTape() as tape:
                prediction = model(ecg)
                loss = mse(score, prediction)
            # Backward
            grad = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grad, model.trainable_weights))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argparse Tutorial')
    parser.add_argument('--epoch', type=int, default=100)
    args = parser.parse_args()

    train_args = {
        'kernel_sizes':KERNEL_SIZE,
        'dimensions':DIMENSION,
        'strides':STRIDE,
        'dropout_rate':DROPOUT_RATE,
    }
    train_model(args.epoch, LEARNING_RATE, **train_args)