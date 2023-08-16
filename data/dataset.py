import pandas as pd
import tensorflow as tf
from config import RAW_DATA_DIR, CSV_PAIR_ROOT

def getDataset(validation_rate):
    data_len = len(pd.read_csv(CSV_PAIR_ROOT))
    threshold = int(data_len*(1-validation_rate))
    
    train_set = tf.data.Dataset.from_generator(
        getDatasetGenerator,
        output_signature=(tf.TensorSpec(shape=(5000, 12), dtype=tf.float32), 
                          tf.TensorSpec(shape=(), dtype=tf.float32)),
        args=(0, threshold)
    )
    test_set = tf.data.Dataset.from_generator(
        getDatasetGenerator,
        output_signature=(tf.TensorSpec(shape=(5000, 12), dtype=tf.float32), 
                          tf.TensorSpec(shape=(), dtype=tf.float32)),
        args=(threshold, data_len)
    )    
    return train_set, test_set


def getDatasetGenerator(start, end):
    csv_data = pd.read_csv(CSV_PAIR_ROOT)
    csv_data = csv_data.iloc[start:end]
    
    for i in range(len(csv_data)):
        data_root = f"{RAW_DATA_DIR}/{csv_data.iloc[i]['fname']}"
        ecg_data = pd.read_csv(data_root).to_numpy()
        score = csv_data.iloc[i]['score']
        
        yield ecg_data, score
        