import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")


class FeatureSelection:
    def __init__(self, train_path: str, test_path: str, nan_threshold: float, corr_threshold: float, variance_threshold: float):
        '''

        train path - path to the train set
        test path - path to the test set
        nan_threhols - if feature has more than *% of data nan, then the column will be deleted
        corr_thresh - delete the column with high correlation
        variance threhsold - delete the columns with quasi constant columns

        '''
        assert nan_threshold >= 0 and nan_threshold <= 100, f'choose number between 0 and 100'
        assert corr_threshold >= 0 and corr_threshold <= 1, 'choose number between 0 and 1'
        assert variance_threshold >= 0 and variance_threshold <= 1, 'choose number 0 and 1'
        assert train_path.endswith('csv') and test_path.endswith(
            'csv'), 'we expect csv file'

        train = pd.read_csv(train_path)
        self.train_target_id = train[['TARGET', 'ID']]
        self.train = train[[
            i for i in train.columns if i != 'TARGET' and i != 'ID']]
        self.test = pd.read_csv(test_path, index_col=[0])
        self.nan_threshold = nan_threshold
        self.corr_threshold = corr_threshold
        self.variance_threshold = variance_threshold
        self.blacklist = ['FEATURE_33']

    def remove_nan_cols(self):
        def get_nan_percent(df):
            columns = list(df.columns)
            num_rows = df.shape[0]
            nan_list = []
            for col in columns:
                nan_percent = df[col].isna().sum()
                nan_list.append((nan_percent / num_rows) * 100)
            df_nan_count = pd.DataFrame(list(zip(columns, nan_list)), columns=[
                                        'column_name', 'nan_percent'])
            return df_nan_count

        df_nan = get_nan_percent(self.train)
        blacklist = df_nan.loc[df_nan.nan_percent >=
                               self.nan_threshold].column_name.tolist()
        self.blacklist.extend(blacklist)
        self.train = self.remove_cols(self.train)

    def remove_constant_cols(self):
        def get_constant_cols(df):
            one_unique = []
            for col in df.columns:
                unique_values = df[col].value_counts().shape[0]
                if unique_values == 1:
                    one_unique.append(col)
            return one_unique
        unique_cols = get_constant_cols(self.train)
        self.blacklist.extend(unique_cols)
        self.train = self.remove_cols(self.train)

    def remove_quasiconst_cols(self):
        df_num = self.get_numerical_dataframe(self.train)
        quasiModel = VarianceThreshold(threshold=self.variance_threshold)
        quasiModel.fit(df_num)
        quasiArr = quasiModel.get_support()
        quasiCols = [
            col for col in df_num.columns if col not in df_num.columns[quasiArr]]
        self.blacklist.extend(quasiCols)
        self.train = self.remove_cols(self.train)

    def remove_duplicated_cols(self):
        df_T = self.train.T
        new_df = df_T.drop_duplicates(keep='first').T
        duplicated_cols = [
            dup_col for dup_col in self.train.columns if dup_col not in new_df.columns]
        self.blacklist.extend(duplicated_cols)
        self.train = self.remove_cols(self.train)

    def remove_correlated_cols(self):
        df_num = self.get_numerical_dataframe(self.train)
        correlation_matrix = df_num.corr()

        corr_features = []
        num_cols = df_num.shape[1]

        corr_features = set()
        for i in range(0, num_cols):
            for j in range(0, i):
                if abs(correlation_matrix.iloc[i, j]) >= self.corr_threshold:
                    corr_features.add(correlation_matrix.columns[i])
        self.blacklist.extend(list(corr_features))
        self.train = self.remove_cols(self.train)

    def fillna_object_cols(self):
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        self.train['FEATURE_106'] = imp.fit_transform(
            self.train['FEATURE_106'].to_numpy().reshape(-1, 1))
        self.test['FEATURE_106'] = imp.transform(
            self.test['FEATURE_106'].to_numpy().reshape(-1, 1))
        imp = SimpleImputer(missing_values=np.nan,
                            strategy='constant', fill_value='Category_3')
        self.train['FEATURE_300'] = imp.fit_transform(
            self.train['FEATURE_300'].to_numpy().reshape(-1, 1))
        self.test['FEATURE_300'] = imp.transform(
            self.test['FEATURE_300'].to_numpy().reshape(-1, 1))

    def get_numerical_dataframe(self, df):
        df_objects = df.select_dtypes(include='object')
        nonnumerical_features = df_objects.columns.tolist()

        numerical_df = df[[
            i for i in df.columns if i not in nonnumerical_features]]
        return numerical_df

    def remove_cols(self, df):
        df = df[[i for i in df.columns if i not in self.blacklist]]
        return df

    def __call__(self):
        print(
            f'Before feature selection, train shape = {self.train.shape}, test shape = {self.test.shape}')
        print()
        self.remove_nan_cols()
        print(
            f'nan cols were removed, length of blacklist = {len(self.blacklist)}')
        self.remove_constant_cols()

        print(
            f'constant columns were removed, length of blacklist = {len(self.blacklist)}')
        self.remove_quasiconst_cols()

        print(
            f'quasi-constant columns were removed, length of blacklist = {len(self.blacklist)}')
        self.remove_duplicated_cols()
        print(
            f'duplicated columns were removed, length of blacklist = {len(self.blacklist)}')

        self.remove_correlated_cols()
        print(
            f'correlated columns were removed, length of blacklist = {len(self.blacklist)}')

        self.fillna_object_cols()
        print(
            f'nan values in object columns were filled')

        self.train = self.train.join(self.train_target_id, how='left')
        self.test = self.remove_cols(self.test)
        self.train.to_csv('./utils/train_features_selected.csv', index=False)
        self.test.to_csv('./utils/test_features_selected.csv', index=False)
        print()
        print(
            f'After feature selection, train shape = {self.train.shape}, test shape = {self.test.shape}')
        return self.train, self.test
