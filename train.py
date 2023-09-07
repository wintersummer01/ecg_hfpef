import argparse
import sys
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score

from utils import setOutputDir, toHfpefScore
from model.resnet import ecg2Hfpef
# from model.CNN import ecg_CNN
from data.dataset import getCritDataset
from config import KERNEL_SIZE, DIMENSION, STRIDE, DROPOUT_RATE, LEARNING_RATE

def train_model(num_epoch, lr, print_out, train_tag="default", **model_args):
    path = setOutputDir(train_tag, print_out)
    
    model = ecg2Hfpef(**model_args)
#     model = tf.keras.models.load_model('./logs/binary_resnet_0/model_2')
    model.load_weights('./logs/binary_resnet_0/weight_2')
    train_ds, test_ds = getCritDataset(training=True)

    optimizer = tf.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    summary_writer = tf.summary.create_file_writer(path+"/values")
    
    min_rms = 9999
    max_R2 = 0
    for i in range(num_epoch):
        # For history
        print(f"\nEpoch {i+1} training")
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
            prob = tf.math.sigmoid(pred)
            train_mse.append(np.mean((crit - prob)**2))
            losses.append(loss)
            
        # Print train process
        avg_loss = np.mean(losses)
        train_rms = np.mean(train_mse)**0.5
        print(f"Training finished")
        print(f"Mean Loss : {avg_loss:.5f}")
        print(f"RMS of each probability : {train_rms:.5f}")
        print(f"Calculating R squared score in validation set")
        
        # Validation Process
        for ecg, score in tqdm(test_ds):
            pred = model(ecg)
            prob = tf.math.sigmoid(pred)
            pred_score = toHfpefScore(prob)
            prediction.append(pred_score)
            ground_truth.append(score)
            test_mse.append(np.mean((score - pred_score)**2))
            
        prediction = np.hstack(prediction)
        ground_truth = np.vstack(ground_truth).reshape(-1)
        R2_score = r2_score(ground_truth, prediction)
        # Print validation process
        test_rms = np.mean(test_mse)**0.5
        print(f"RMS of hfpef score : {test_rms:.5f}")
        print(f"R2 Score : {R2_score:.5f}\n")
        
        # Save Weight
        if test_rms < min_rms:
            model.save(path + f'/model_{i}')
            model.save_weights(path + f'/weight_{i}')
            min_rms = test_rms
        elif R2_score > max_R2:
            model.save(path + f'/model_{i}')
            model.save_weights(path + f'/weight_{i}')
            max_R2 = R2_score
            
        
        # Write in Summary writer
        with summary_writer.as_default():
            tf.summary.scalar('average loss', avg_loss, i+1)
            tf.summary.scalar('train RMS', train_rms, i+1)
            tf.summary.scalar('test RMS', test_rms, i+1)
            tf.summary.scalar('R2 Score', R2_score, i+1)
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--print_out', action='store_true')
    args = parser.parse_args()

    train_tag = "binary_resnet"
    model_args = {
        'kernel_sizes':KERNEL_SIZE,
        'dimensions':DIMENSION,
        'strides':STRIDE,
        'dropout_rate':DROPOUT_RATE
    }
    train_model(args.epoch, LEARNING_RATE, \
                args.print_out, train_tag=train_tag, **model_args)
    
    
    
    
    