import pickle
import os 
import numpy as np


def get_fold_data(data_dir, fold_data, dataType, labelEncoder, method = 0):
    X = list()#np.empty(len(fold_data))
    y = list()#np.empty(len(fold_data))
        
    maxRow = -1
    for i, fname in enumerate(fold_data.get(dataType)):
        # each file contains a named tupple
        # 'patient_id','seizure_type', 'data'
        seizure = pickle.load(open(os.path.join(data_dir, fname), "rb"))
        
        if(method == 0):
            #Sum rows method
            X.append(np.sum(seizure.data, axis=0))
        elif(method == 1):
            # Pad with rows of zeros method
            if(seizure.data.shape[0] > maxRow):
                maxRow = seizure.data.shape[0]
            X.append(seizure.data)
        
        y.append(seizure.seizure_type)
    if(method == 1):
        for i in range(len(X)):
            X[i] = np.pad(X[i], ((0, len(X[i]) -maxRow), (0,0)))
            print(X[i].shape)
    y = labelEncoder.transform(y)
    return X, y
