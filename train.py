import argparse
import sys
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score

from utils import setOutputDir
from model.resnet import ecg2Hfpef
from data.dataset import getDataset
from config import KERNEL_SIZE, DIMENSION, STRIDE, DROPOUT_RATE, \
LEARNING_RATE, BATCH_SIZE, VALIDATION_RATE

def train_model(num_epoch, lr, val_rate, batch_size, print_out, train_tag="default", **train_args):
    path = setOutputDir(train_tag, print_out)
    
    model = ecg2Hfpef(**train_args)
    train_ds, test_ds = getDataset(val_rate, batch_size)

    optimizer = tf.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mse = tf.losses.mean_squared_error
    
    train_mse = []
    losses = []
    for i in range(num_epoch):
        print(f"Epoch {i+1} training")
        # Training Process
        for ecg, score in tqdm(train_ds, miniters=100, maxinterval=100):
            # Forward
            with tf.GradientTape() as tape:
                prediction = model(ecg, training=True)
                loss = loss_fn(score, prediction)
            # Backward
            grad = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grad, model.trainable_weights))
            # History
            train_mse.append(mse(score, prediction).numpy())
            losses.append(loss.numpy())
        
        # Save Weight
        model.save_weights(path + f'/weight_{i}')
        print(f"Training finished. RMS: {np.mean(np.hstack(train_mse))**0.5:.2f}")
        print(f"Mean Loss : {np.mean(np.hstack(losses))}")
        print(f"Calculating R squared score in validation set")
        
        # Validation Process
        prediction = []
        ground_truth = []
        for ecg, score in tqdm(test_ds, miniters=10, maxinterval=100):
            pred = model(ecg)
            prediction.append(np.argmax(pred, axis=-1))
            ground_truth.append(score.numpy().reshape(-1))
        prediction, ground_truth = np.hstack(prediction), np.hstack(ground_truth)
        R2_score = r2_score(ground_truth, prediction)
        print(f"R2 Score : {R2_score}\n")
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--print_out', action='store_true')
    args = parser.parse_args()

    train_tag = "cross_entropy"
    train_args = {
        'kernel_sizes':KERNEL_SIZE,
        'dimensions':DIMENSION,
        'strides':STRIDE,
        'dropout_rate':DROPOUT_RATE
    }
    train_model(args.epoch, LEARNING_RATE, VALIDATION_RATE, BATCH_SIZE,\
                args.print_out, train_tag=train_tag, **train_args)
    
    
    
    
    