import argparse
import os
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt 
from abc import ABC, abstractmethod
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class SeizureClassification(ABC):
    def run(self, cross_val_file, data_dir):
        sz = pickle.load(open(cross_val_file, "rb"))
    
        szr_type_list = ['TNSZ', 'SPSZ', 'ABSZ', 'TCSZ', 'CPSZ', 'GNSZ', 'FNSZ']
        le = preprocessing.LabelEncoder()
        le.fit(szr_type_list)
        
        results = []
        originalLabels = []
        predictedLabels = []
        # Iterate through the folds
        for i, fold_data in enumerate(sz.values()):
        
            # fold_data is a dictionary with train and val keys
            # Each contains a list of name of files
        
            X_train, y_train = getFoldData(data_dir, fold_data, "train", le)
            X_test, y_test = getFoldData(data_dir, fold_data, "val", le)
            model = self._generateModel()

            clf = model.fit(X_train, y_train)
            predicted = model.predict(X_test)
            originalLabels.extend(y_test)
            predictedLabels.extend(predicted)
            score = clf.score(X_test, y_test)
            results.append(score)
        
            print("Fold number ",i, " completed. Score: ", score)
            self._printFoldResults(clf)
        print("Classification report for classifier %s:\n%s\n"
      % (model, metrics.classification_report(originalLabels, predictedLabels)))
        
        print("Avg result: ", np.mean(results) )
    
    @abstractmethod
    def _generateModel(self):
        pass
    def _printFoldResults(self, estimator):
        return
        
            
class KNeighboursClassification(SeizureClassification):
    def _generateModel(self):
        return KNeighborsClassifier()
    

class KNeighboursRandomSearchClassification(SeizureClassification):
    def _generateModel(self):
        hyperparameters = dict(leaf_size=list(range(1,50)), n_neighbors=list(range(1,30)), p=[1,2])
        clf = KNeighborsClassifier()
        return RandomizedSearchCV(clf, hyperparameters)
    
    def _printFoldResults(self, estimator):
        #Print The value of best Hyperparameters
        print('Best leaf_size:', estimator.best_estimator_.get_params()['leaf_size'])
        print('Best p:', estimator.best_estimator_.get_params()['p'])
        print('Best n_neighbors:', estimator.best_estimator_.get_params()['n_neighbors'])

class DecisionTreeClassification(SeizureClassification):

    def _generateModel(self):
        return DecisionTreeClassifier()

class RandomForestClassification(SeizureClassification):
    def _generateModel(self):
        return RandomForestClassifier()

def getFoldData(data_dir, fold_data, dataType, labelEncoder, method = 0):
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
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument("-c","--cross_val_file", help="Pkl cross validation file")
    parser.add_argument("-d", "--data_dir", help="Folder containing all the preprocessed data")
    args = parser.parse_args()
    cross_val_file = args.cross_val_file
    data_dir = args.data_dir
    
    a = DecisionTreeClassification()
    a.run(cross_val_file, data_dir)
   

#python3 model.py -c ./data_preparation/cv_split_3_fold_patient_wise_v1.5.2.pkl -d "/media/david/Extreme SSD/Machine Learning/raw_data/v1.5.2/fft_with_time_freq_corr/fft_seizures_wl1_ws_0.5_sf_250_fft_min_1_fft_max_12"
 #disp = metrics.plot_confusion_matrix(model, X_test, y_test, display_labels=le.classes_)
    #plt.show() 
    #disp.figure_.suptitle("Confusion Matrix")
    #print("Confusion matrix:\n%s" % disp.confusion_matrix)
   #python3 model.py -c ../seizure-type-classification-tuh/data_preparation/cv_split_3_fold_patient_wise_v1.5.2.pkl -d "/home/david/Documents/Machine Learning/raw_data/fft_seizures_wl1_ws_0.5_sf_250_fft_min_1_fft_max_12"
