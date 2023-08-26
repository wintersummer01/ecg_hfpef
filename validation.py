from tqdm import tqdm
import numpy as np

import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error 

from data.dataset import getDataset
from model.resnet import ecg2Hfpef
from config import KERNEL_SIZE, DIMENSION, STRIDE, DROPOUT_RATE, LOG_ROOT

def valid_model(tag, weight_no, **model_args):
    model_path = LOG_ROOT + '/' + tag + f'/weight_{weight_no}'
    model = ecg2Hfpef(**model_args)
    model.load_weights(model_path)
    test_ds = getDataset(mode='test')
    
    prediction = []
    ground_truth = []
    for ecg, score in tqdm(test_ds, miniters=10, maxinterval=100):
        pred = model(ecg)
        prediction.append(np.argmax(pred, axis=-1))
        ground_truth.append(score.numpy().reshape(-1))
    prediction, ground_truth = np.hstack(prediction), np.hstack(ground_truth)
    R2_score = r2_score(ground_truth, prediction)
    mse = mean_squared_error(ground_truth, prediction)
    print(f"R2 Score : {R2_score}\n")
    print(f"RMS : {mse**0.5:.2f}\n")
    
    
if __name__ == '__main__':
    tag = "cross_entropy_0"
    weight_no = 4
    model_args = {
        'kernel_sizes':KERNEL_SIZE,
        'dimensions':DIMENSION,
        'strides':STRIDE,
        'dropout_rate':DROPOUT_RATE
    }
    valid_model(tag, weight_no, **model_args)