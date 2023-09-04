from tqdm import tqdm
import numpy as np

import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error 

from data.dataset import getCritDataset
from model.resnet import ecg2Hfpef
from config import KERNEL_SIZE, DIMENSION, STRIDE, DROPOUT_RATE, LOG_DIR

def valid_model(tag, weight_no, **model_args):
    model_path = LOG_DIR + '/' + tag + f'/weight_{weight_no}'
    model = ecg2Hfpef(**model_args)
    model.load_weights(model_path)
    test_ds = getCritDataset(training=False)
    prediction = []
    ground_truth = []
    
    # Validation Process
    for ecg, score in tqdm(test_ds):
        pred = model(ecg)
        pred_score = toHfpefScore(pred)
        prediction.append(pred_score)
        ground_truth.append(score)
    prediction = np.hstack(prediction)
    ground_truth = np.vstack(ground_truth).reshape(-1)
    R2_score = r2_score(ground_truth, prediction)
    
    # Print validation result
    print(f"R2 Score : {R2_score}\n")
    
    
if __name__ == '__main__':
    tag = "binary_entropy_0"
    weight_no = 4
    model_args = {
        'kernel_sizes':KERNEL_SIZE,
        'dimensions':DIMENSION,
        'strides':STRIDE,
        'dropout_rate':DROPOUT_RATE
    }
    valid_model(tag, weight_no, **model_args)