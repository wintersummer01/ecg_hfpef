import numpy as np
from config import BATCH_SIZE

def makeRandomDataset(shape=(5000, 12), num_batches=5, batch_size=BATCH_SIZE):
    num_samples, dim = shape
    dataset = []
    for _ in range(num_batches):
        dataset.append(
            (np.random.random([batch_size, num_samples, dim]), np.random.random())
            )
    return dataset
