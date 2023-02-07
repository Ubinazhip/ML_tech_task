from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")


def one_hot_encoder(train, val, test, save_artefacts):
    one_hot_feature300 = OneHotEncoder(
        drop='first', handle_unknown='error')
    arr_train = one_hot_feature300.fit_transform(
        train['FEATURE_300'][:, None]).toarray()
    arr_val = one_hot_feature300.transform(
        val['FEATURE_300'][:, None]).toarray()
    arr_test = one_hot_feature300.transform(
        test['FEATURE_300'][:, None]).toarray()

    cols_name = ['FEATURE_300_0', 'FEATURE_300_1']
    df_feature300 = pd.DataFrame(arr_train, columns=cols_name, dtype=int)
    train = train.join(df_feature300)

    df_feature300 = pd.DataFrame(arr_val, columns=cols_name, dtype=int)
    val = val.join(df_feature300)

    df_feature300 = pd.DataFrame(arr_test, columns=cols_name, dtype=int)
    test = test.join(df_feature300)

    one_hot_feature106 = OneHotEncoder(
        drop='first', handle_unknown='error')
    arr_train = one_hot_feature106.fit_transform(
        train['FEATURE_106'][:, None]).toarray()
    arr_val = one_hot_feature106.transform(
        val['FEATURE_106'][:, None]).toarray()
    arr_test = one_hot_feature106.transform(
        test['FEATURE_106'][:, None]).toarray()

    train['FEATURE_106'] = arr_train
    val['FEATURE_106'] = arr_val
    test['FEATURE_106'] = arr_test

    one_hot_feature_27 = OneHotEncoder(
        drop='first', handle_unknown='error')
    arr_train = one_hot_feature_27.fit_transform(
        train['FEATURE_27'][:, None]).toarray()
    arr_val = one_hot_feature_27.transform(
        val['FEATURE_27'][:, None]).toarray()
    arr_test = one_hot_feature_27.transform(
        test['FEATURE_27'][:, None]).toarray()

    cols_name = ['FEATURE_27_0', 'FEATURE_27_1']
    df_feature27 = pd.DataFrame(arr_train, columns=cols_name, dtype=int)
    train = train.join(df_feature27)
    df_feature27 = pd.DataFrame(arr_val, columns=cols_name, dtype=int)
    val = val.join(df_feature27)
    df_feature27 = pd.DataFrame(arr_test, columns=cols_name, dtype=int)
    test = test.join(df_feature27)

    train = train.drop('FEATURE_300', axis=1)
    train = train.drop('FEATURE_27', axis=1)
    val = val.drop('FEATURE_300', axis=1)
    val = val.drop('FEATURE_27', axis=1)
    test = test.drop('FEATURE_300', axis=1)
    test = test.drop('FEATURE_27', axis=1)

    if save_artefacts:
        pickle.dump(one_hot_feature300, open(
            f'./utils/one_hot_feature300.pkl', 'wb'))
        pickle.dump(one_hot_feature106, open(
            f'./utils/one_hot_feature106.pkl', 'wb'))
        pickle.dump(one_hot_feature_27, open(
            f'./utils/one_hot_feature27.pkl', 'wb'))
    return train, val, test
