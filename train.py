import argparse
import sys
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score

from utils import setOutputDir, toHfpefScore
from model.resnet import ecg2Hfpef
from data.dataset import getCritDataset
from config import KERNEL_SIZE, DIMENSION, STRIDE, DROPOUT_RATE, LEARNING_RATE

def train_model(num_epoch, lr, print_out, train_tag="default", **model_args):
    path = setOutputDir(train_tag, print_out)
    
    model = ecg2Hfpef(**model_args)
    train_ds, test_ds = getCritDataset(training=True)

    optimizer = tf.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    for i in range(num_epoch):
        # For history
        print(f"Epoch {i+1} training")
        train_mse = []
        losses = []
        test_mse = []
        prediction = []
        ground_truth = []
        
        # Training Process
        for ecg, crit in tqdm(train_ds):
            # Forward
            with tf.GradientTape() as tape:
                pred = model(ecg, training=True)
                loss = loss_fn(crit, pred)
            # Backward
            grad = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grad, model.trainable_weights))
            # History
            train_mse.append(np.mean((crit - pred)**2))
            losses.append(loss)
            
        # Print train process
        print(f"Training finished")
        print(f"RMS of each probability : {np.mean(train_mse)**0.5:.2f}")
        print(f"Mean Loss : {np.mean(losses)}\n")
        print(f"Calculating R squared score in validation set")
        
        # Save Weight
        model.save_weights(path + f'/weight_{i}')      
        
        # Validation Process
        for ecg, score in tqdm(test_ds):
            pred = model(ecg)
            pred_score = toHfpefScore(pred)
            prediction.append(pred_score)
            ground_truth.append(score)
            test_mse.append(np.mean((score - pred_score)**2))
        prediction = np.hstack(prediction)
        ground_truth = np.vstack(ground_truth).reshape(-1)
        R2_score = r2_score(ground_truth, prediction)
        # Print validation process
        print(f"R2 Score : {R2_score}\n")
        print(f"RMS of score : {np.mean(test_mse)**0.5:.2f}")
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--print_out', action='store_true')
    args = parser.parse_args()

    train_tag = "binary_entropy"
    model_args = {
        'kernel_sizes':KERNEL_SIZE,
        'dimensions':DIMENSION,
        'strides':STRIDE,
        'dropout_rate':DROPOUT_RATE
    }
    train_model(args.epoch, LEARNING_RATE, \
                args.print_out, train_tag=train_tag, **model_args)
    
    
    
    
    