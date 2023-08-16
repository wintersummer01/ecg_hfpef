import argparse
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score


from model.resnet import ecg2Hfpef
from data.dataset import getDataset
from config import KERNEL_SIZE, DIMENSION, STRIDE, \
DROPOUT_RATE, LEARNING_RATE, BATCH_SIZE, VALIDATION_RATE

def train_model(num_epoch, lr, val_rate, batch_size, **train_args):
    model = ecg2Hfpef(**train_args)
    train_ds, test_ds = getDataset(val_rate)

    optimizer = tf.optimizers.Adam(learning_rate=lr)
    mse = tf.losses.mean_squared_error
    
    train_mse = []
    for i in range(num_epoch):
        print(f"Epoch {i+1} training")
        for ecg, score in tqdm(train_ds.batch(batch_size)):
            # Forward
            with tf.GradientTape() as tape:
                prediction = model(ecg, training=True)
                loss = mse(score, prediction)
            # Backward
            grad = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grad, model.trainable_weights))
            # History
            train_mse.append(loss)
        print(f"Training finished. RMS: {np.mean(train_mse)**0.5:.2f}")
        print(f"Calculating R squared score in validation set")
        
        ground_truth = []
        prediction = []
        for ecg, score in tqdm(test_ds.batch(1)):
            ground_truth.append(score)
            pred = model(ecg)
            prediction.append(float(pred))
        R2_score = r2_score(prediction, ground_truth)
        print(f"R2 Score : {R2_score}\n")
        model.save_weights(f'logs/weight_{i}')
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=5)
    args = parser.parse_args()

    train_args = {
        'kernel_sizes':KERNEL_SIZE,
        'dimensions':DIMENSION,
        'strides':STRIDE,
        'dropout_rate':DROPOUT_RATE
    }
    train_model(args.epoch, LEARNING_RATE, VALIDATION_RATE, \
                BATCH_SIZE, **train_args)