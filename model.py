import argparse
import os
import pickle
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, metrics
import xgboost as xgb
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class SeizureClassification(ABC):
    def run(self, cross_val_file, data_dir):
        sz = pickle.load(open(cross_val_file, "rb"))

        szr_type_list = ['TNSZ', 'SPSZ', 'ABSZ', 'TCSZ', 'CPSZ', 'GNSZ', 'FNSZ']
        le = preprocessing.LabelEncoder()
        le.fit(szr_type_list)

        results = []
        original_labels = []
        predicted_labels = []
        # Iterate through the folds
        for i, fold_data in enumerate(sz.values()):
            # fold_data is a dictionary with train and val keys
            # Each contains a list of name of files

            X_train, y_train = get_fold_data(data_dir, fold_data, "train", le)
            X_test, y_test = get_fold_data(data_dir, fold_data, "val", le)
            model = self._generate_model()

            clf = model.fit(X_train, y_train)
            predicted = model.predict(X_test)
            original_labels.extend(y_test)
            predicted_labels.extend(predicted)
            score = clf.score(X_test, y_test)
            results.append(score)
            print("Fold number ", i, " completed. Score: ", '{:.2f}'.format(score))
        print("Classification report for classifier %s:\n%s\n"
              % ("Test", metrics.classification_report(original_labels, predicted_labels)))

        print("Avg result: ", '{:.2f}'.format(np.mean(results)))

    @abstractmethod
    def _generate_model(self):
        pass


class KNeighboursClassification(SeizureClassification):
    def _generate_model(self):
        return KNeighborsClassifier()


class SGDClassification(SeizureClassification):
    def _generate_model(self):
        # Always scale the input. The most convenient way is to use a pipeline.
        return make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))


class XGBoostClassification(SeizureClassification):
    def _generate_model(self):
        return xgb.XGBClassifier()

    def run(self, cross_val_file, data_dir):
        sz = pickle.load(open(cross_val_file, "rb"))

        szr_type_list = ['TNSZ', 'SPSZ', 'ABSZ', 'TCSZ', 'CPSZ', 'GNSZ', 'FNSZ']
        le = preprocessing.LabelEncoder()
        le.fit(szr_type_list)

        # setup parameters for xgboost
        # https://xgboost.readthedocs.io/en/latest/parameter.html
        param = {}
        # use softmax multi-class classification
        param['objective'] = 'multi:softmax'
        # scale weight of positive examples
        param['eta'] = 0.1
        param['max_depth'] = 6
        param['nthread'] = 4
        param['num_class'] = len(szr_type_list)

        error_rates = []
        # Iterate through the folds
        for i, fold_data in enumerate(sz.values()):
            # fold_data is a dictionary with train and val keys
            # Each contains a list of name of files

            X_train, y_train = get_fold_data(data_dir, fold_data, "train", le)
            X_test, y_test = get_fold_data(data_dir, fold_data, "val", le)
            X_train = np.array(X_train)
            X_test = np.array(X_test)

            # Convert training instances and testing instances into DMatrices
            xg_train = xgb.DMatrix(X_train, label=y_train)
            xg_test = xgb.DMatrix(X_test, label=y_test)

            watchlist = [(xg_train, 'train'), (xg_test, 'test')]
            num_round = 5
            bst = xgb.train(param, xg_train, num_round, watchlist)
            # get prediction
            pred = bst.predict(xg_test)
            error_rate = np.sum(pred != y_test) / y_test.shape[0]
            print("Fold number ", i, " completed. Error rate: ", '{:.2f}'.format(error_rate))
            error_rates.append(error_rate)
        print("Avg error rate using softmax: ", '{:.2f}'.format(np.mean(error_rates)))


def get_fold_data(data_dir, fold_data, dataType, labelEncoder):
    X = list()  # np.empty(len(fold_data))
    y = list()  # np.empty(len(fold_data))

    for i, fname in enumerate(fold_data.get(dataType)):
        # each file contains a named tupple
        # 'patient_id','seizure_type', 'data'
        seizure = pickle.load(open(os.path.join(data_dir, fname), "rb"))
        X.append(np.sum(seizure.data, axis=0))

        y.append(seizure.seizure_type)
    y = labelEncoder.transform(y)
    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument("-c", "--cross_val_file", help="Pkl cross validation file")
    parser.add_argument("-d", "--data_dir", help="Folder containing all the preprocessed data")
    args = parser.parse_args()
    cross_val_file = args.cross_val_file
    data_dir = args.data_dir

    a = KNeighboursClassification()
    a.run(cross_val_file, data_dir)

    b = SGDClassification()
    b.run(cross_val_file, data_dir)

    c = XGBoostClassification()
    c.run(cross_val_file, data_dir)

# python3 model.py -c ./data_preparation/cv_split_3_fold_patient_wise_v1.5.2.pkl -d "/media/david/Extreme SSD/Machine Learning/raw_data/v1.5.2/fft_with_time_freq_corr/fft_seizures_wl1_ws_0.5_sf_250_fft_min_1_fft_max_12"
