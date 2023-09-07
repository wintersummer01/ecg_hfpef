from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error 

from utils import toHfpefScore
from data.dataset import getCritDataset
from model.resnet import ecg2Hfpef
from model.CNN import ecg_CNN
from config import KERNEL_SIZE, DIMENSION, STRIDE, DROPOUT_RATE, LOG_DIR



def valid_model(model, path, dataset, ds_type):
    probs = []
    crits = []
    
    # Compute prediction
    for ecg, crit in tqdm(dataset):
        pred = model(ecg)
        prob = tf.math.sigmoid(pred)
        probs.append(prob)
        crits.append(crit)
    probs = np.vstack(probs)
    crits = np.vstack(crits)
    pred_score = toHfpefScore(probs)
    gt_score = toHfpefScore(crits)
    
    # Scatter Plot (crits)
    crit_key = ['BMI', 'afib', 'PASP', 'age', 'EE']
    for i in range(5):
        plt.scatter(crits[:,i], probs[:,i])
        plt.title(f'{crit_key[i]} probability scatter plot ({ds_type} data)')
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.savefig(path + f'/scatter_{crit_key[i]}_{ds_type}.png')
        plt.cla()
        
    # Scatter Plot (score)
    plt.scatter(gt_score, pred_score)
    plt.title(f'H2FpEF score prediction scatter plot ({ds_type} data)')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.savefig(path + f'/scatter_score_{ds_type}.png')
    plt.cla()
    
    # Measurements
    prob_rms = np.mean((crits - probs)**2)**0.5
    score_rms = np.mean((gt_score-pred_score)**2)**0.5
    R2_score = r2_score(gt_score, pred_score)
    print(f"Probability RMS of {ds_type} dataset : {prob_rms}")
    print(f"Score RMS of {ds_type} dataset : {score_rms}")
    print(f"R2 Score : {R2_score}")
    
    
if __name__ == '__main__':
    tag = "binary_resnet_1"
    weight_no = 51
    model_args = {
        'kernel_sizes':KERNEL_SIZE,
        'dimensions':DIMENSION,
        'strides':STRIDE,
        'dropout_rate':DROPOUT_RATE
    }
    
    model_path = LOG_DIR + tag
    model = ecg2Hfpef(**model_args)
    model.load_weights("logs/binary_resnet_1/weight_51")
#     model = ecg_CNN()
#     model.load_weights("./logs/binary_entropy_3/weight_29")
#     model = tf.keras.models.load_model("./logs/binary_entropy_3/model_29")
    
    train_ds, test_ds = getCritDataset(training=False)
    
    # valid_model(model_path, train_ds, 'train')
    valid_model(model, model_path, test_ds, 'test')