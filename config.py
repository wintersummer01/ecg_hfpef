RAW_DATA_DIR = '/home/ubuntu/ecg_mount/220927_SevMUSE_EKG_waveform'
CSV_PAIR_ROOT = '/home/ubuntu/ecg_hfpef/data/fname_score_pair.csv'
LOG_ROOT = '/home/ubuntu/ecg_hfpef/logs/'

LEARNING_RATE = 0.001
VALIDATION_RATE = 0.1

BATCH_SIZE = 64
KERNEL_SIZE = [17, 13, 9, 7, 5]

# Assuming input dimension (5000, 12)
DIMENSION = [12, 64, 128, 196, 256, 512]
STRIDE = [2, 4, 5, 5, 5]
DROPOUT_RATE = 0.5
# data size = (5000, 2500, 625, 125, 25, 5)