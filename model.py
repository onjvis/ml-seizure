import argparse
import os
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt 

def getFoldData(fold_data, dataType, labelEncoder):
    X = list()#np.empty(len(fold_data))
    y = list()#np.empty(len(fold_data))
        
    for i, fname in enumerate(fold_data.get(dataType)):
        # each file contains a named tupple
        # 'patient_id','seizure_type', 'data'
        seizure = pickle.load(open(os.path.join(data_dir, fname), "rb"))
        #if(not isinstance(seizure.data, np.ndarray) or seizure.data.ndim != 2):
        #    print("HELLO")
        X.append(np.sum(seizure.data, axis=0))
        #if(np.array(X_train).ndim > 1):
        #    print("DMIM")
        
        y.append(seizure.seizure_type)
    y = labelEncoder.transform(y)
    return X, y
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument("-c","--cross_val_file", help="Pkl cross validation file")
    parser.add_argument("-d", "--data_dir", help="Folder containing all the preprocessed data")
    args = parser.parse_args()
    cross_val_file = args.cross_val_file
    data_dir = args.data_dir
    sz = pickle.load(open(cross_val_file, "rb"))
    szr_type_list = ['TNSZ', 'SPSZ', 'ABSZ', 'TCSZ', 'CPSZ', 'GNSZ', 'FNSZ']
    
    le = preprocessing.LabelEncoder()
    le.fit(szr_type_list)
    

    # Iterate through the folds
    for fold_data in sz.values():
    
        # fold_data is a dictionary with train and val keys
        # Each contains a list of name of files
       
        X_train, y_train = getFoldData(fold_data, "train", le)
        X_test, y_test = getFoldData(fold_data, "val", le)
        neigh = KNeighborsClassifier()

        clf = neigh.fit(X_train, y_train)
        predicted = neigh.predict(X_test)
        print("Classification report for classifier %s:\n%s\n"
      % (neigh, metrics.classification_report(y_test, predicted)))
        disp = metrics.plot_confusion_matrix(neigh, X_test, y_test, display_labels=le.classes_)
        plt.show() 
        disp.figure_.suptitle("Confusion Matrix")
        print("Confusion matrix:\n%s" % disp.confusion_matrix)
        print(clf.score(X_test, y_test))
        #print(y_train[-1])
        #print(neigh.predict([X_train[-1]]))
        #print(neigh.predict_proba([X_train[-1]]))
        exit() # To end in the first run
    
