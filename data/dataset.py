import pandas as pd
import numpy as np
import tensorflow as tf
from config import RAW_DATA_DIR, CSV_PAIR_ROOT, PROC_DATA_DIR, \
VALIDATION_RATE, BATCH_SIZE, CRIT_PAIR_ROOT, DATA_BS

def critDataGen(start, end, training):
    crit_pairs = pd.read_csv(CRIT_PAIR_ROOT)
    for i in range(start, end):
        data_batch = np.load(f"{PROC_DATA_DIR}/batch_{i}.npy")
        label_df = crit_pairs.query(f"batch_no=={i}")
        if training:
            label = label_df[['BMI', 'afib', 'PASP', 'age', 'EE']].to_numpy()
        else:
            label = label_df[['score']].to_numpy()
        
        yield data_batch, label
    

def getCritDataset(validation_rate=VALIDATION_RATE, batch_size=BATCH_SIZE, training=True):
    crit_pairs = pd.read_csv(CRIT_PAIR_ROOT)
    num_batches = len(crit_pairs)//DATA_BS + 1
    threshold = int(num_batches*(1-validation_rate))
    
    train_set = tf.data.Dataset.from_generator(
        critDataGen,
        output_signature=(tf.TensorSpec(shape=(None, 5000, 12), dtype=tf.float32),
                          tf.TensorSpec(shape=(None, 5), dtype=tf.float32)),
        args=(0, threshold, 1)
    )
    test_set = tf.data.Dataset.from_generator(
        critDataGen,
        output_signature=(tf.TensorSpec(shape=(None, 5000, 12), dtype=tf.float32),
                          tf.TensorSpec(shape=(None, 1), dtype=tf.float32)),
        args=(threshold, num_batches, 0)
    )
        
    train_set = train_set.unbatch().shuffle(batch_size*8).batch(batch_size)
    test_set = test_set.unbatch().batch(batch_size)
    
    if training == True:
        return train_set, test_set
    else:
        return test_set



def getDatasetGenerator(start, end):
    csv_data = pd.read_csv(CSV_PAIR_ROOT)
    csv_data = csv_data.iloc[start:end]
    for i in range(len(csv_data)):
        data_root = f"{RAW_DATA_DIR}/{csv_data.iloc[i]['fname']}"
        ecg_data = pd.read_csv(data_root).to_numpy()
        score = csv_data.iloc[i]['score']
        yield ecg_data, (score, )


def getDataset(validation_rate=VALIDATION_RATE, batch_size=BATCH_SIZE, \
               mode='train'):
    
    data_len = len(pd.read_csv(CSV_PAIR_ROOT))
    threshold = int(data_len*(1-validation_rate))
    
    test_set = tf.data.Dataset.from_generator(
        getDatasetGenerator,
        output_signature=(tf.TensorSpec(shape=(5000, 12), dtype=tf.float32), 
                          tf.TensorSpec(shape=(1, ), dtype=tf.float32)),
        args=(threshold, data_len)
    )    
    if mode == 'train':
        train_set = tf.data.Dataset.from_generator(
            getDatasetGenerator,
            output_signature=(tf.TensorSpec(shape=(5000, 12), dtype=tf.float32), 
                              tf.TensorSpec(shape=(1, ), dtype=tf.float32)),
            args=(0, threshold)
        )
        return train_set.batch(batch_size), test_set.batch(batch_size)
    
    elif mode == 'test':
        return test_set.batch(batch_size)
        


        