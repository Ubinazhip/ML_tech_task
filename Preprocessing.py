import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from onehotencoder import one_hot_encoder
from sklearn.model_selection import train_test_split
import warnings
import pickle
import numpy as np
warnings.filterwarnings("ignore")


class Preprocessor:
    def __init__(self, train_path: str, test_path: str, val_size: float, random_state: int, save_artefacts: bool):
        '''

        train_path - path to the train set
        test_path - path to the test set
        val size - size of the validation set, 0 - 1
        random state - random state for train val split
        save artefacts - wether to save one hot encoders, scaler

        '''
        self.full_train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)
        self.val_size = val_size
        self.random_state = random_state
        self.save_artefacts = save_artefacts

    def train_val_split(self):
        train, val, train_labels, val_labels = train_test_split(self.full_train.drop(labels=['TARGET'], axis=1),
                                                                self.full_train['TARGET'],
                                                                test_size=self.val_size,
                                                                random_state=self.random_state)
        train = train.join(train_labels)
        val = val.join(val_labels)
        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)
        return train, val

    def fillna_num_cols(self, train, val, test):
        nan_cols = train.columns[train.isnull().any()].tolist()

        for col in nan_cols:
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            train[col] = imp.fit_transform(
                train[col].to_numpy().reshape(-1, 1))
            val[col] = imp.transform(val[col].to_numpy().reshape(-1, 1))
            test[col] = imp.transform(test[col].to_numpy().reshape(-1, 1))
        return train, val, test

    def standard_scaler(self, train, val, test):
        no_scaling_cols = ['FEATURE_300_0', 'FEATURE_300_1',
                           'FEATURE_27_0', 'FEATURE_27_1',
                           'FEATURE_106', 'ID', 'TARGET']
        df_objects = train.select_dtypes(include='object')

        no_scaling_cols.extend(df_objects.columns)
        numeric_cols = [i for i in train.columns if i not in no_scaling_cols]

        scaler = StandardScaler()
        train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
        val[numeric_cols] = scaler.transform(val[numeric_cols])
        test[numeric_cols] = scaler.transform(test[numeric_cols])
        if self.save_artefacts:
            pickle.dump(scaler, open(f'./utils/feature_scaler.pkl', 'wb'))
        return train, val, test

    def __call__(self):
        train, val = self.train_val_split()
        print(f'train val split has been done, val size = {self.val_size}')

        train, val, test = one_hot_encoder(
            train=train, val=val, test=self.test, save_artefacts=self.save_artefacts)

        print('object features has been one hot encoded ... ')
        train, val, test = self.fillna_num_cols(train, val, test)
        print(f'numerical missing values were filled...')
        train, val, test = self.standard_scaler(train, val, test)
        print(f'numerical features has been scaled ...')
        train.to_csv('./utils/train_preprocessed.csv', index=False)
        val.to_csv('./utils/val_preprocessed.csv', index=False)
        test.to_csv('./utils/test_preprocessed.csv', index=False)
