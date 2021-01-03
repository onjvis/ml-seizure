SZR_CLASSES = ['TNSZ', 'SPSZ', 'ABSZ', 'TCSZ', 'CPSZ', 'GNSZ', 'FNSZ']
EEG_WINDOWS=156 # this number is the average of rows that the eeg dataset used has. To calculate it, get_fold_data from utils was used
EEG_COLUMNS = 660 # 900

BATCH_SIZE = 8 # Batch size 8 seems to be the limit with my machine
EEG_SHAPE = (EEG_WINDOWS,EEG_COLUMNS)
EPOCHS =100

DATASET_MIN = -2.0223328939329814
DATASET_MAX = 19.953151764087092
