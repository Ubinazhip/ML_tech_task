from Preprocessing import Preprocessor
import warnings
import xgboost as xgb
from metric import pfbeta
import pandas as pd
warnings.filterwarnings("ignore")
from feature_selection import FeatureSelection


if __name__ == "__main__":

    selector = FeatureSelection(train_path='TECH_TASK_TRAIN.csv', test_path='TECH_TASK_TEST.csv',
                                nan_threshold=40, corr_threshold=0.8, variance_threshold=0.01)
    train, test = selector()

    preprocessor = Preprocessor(train_path='./utils/train_features_selected.csv',
                                test_path='./utils/test_features_selected.csv',
                                val_size=0.2, random_state=42, save_artefacts=False)

    preprocessor()
    train = pd.read_csv('./utils/train_preprocessed.csv')
    val = pd.read_csv('./utils/val_preprocessed.csv')
    test = pd.read_csv('./utils/test_preprocessed.csv')

    y_train = train['TARGET']
    X_train = train.drop(['ID', 'TARGET', 'FEATURE_121'], axis=1)

    y_val = val['TARGET']
    X_val = val.drop(['ID', 'TARGET', 'FEATURE_121'], axis=1)

    X_test = test.drop(['ID', 'FEATURE_121'], axis=1)

    best_params = {
        'colsample_bytree': 0.5,
        'gamma': 0.25,
        'learning_rate': 0.1,
        'max_depth': 4,
        'reg_lambda': 0,
        'scale_pos_weight': 5,
        'subsample': 0.8
    }

    xgb_cl = xgb.XGBClassifier(**best_params)

    # Fit
    xgb_cl.fit(X_train, y_train)
    val_preds = xgb_cl.predict(X_val)
    train_preds = xgb_cl.predict(X_train)
    test_preds = xgb_cl.predict(X_test)

    pfbeta_val = pfbeta(labels=y_val.to_numpy(), predictions=val_preds)
    pfbeta_train = pfbeta(labels=y_train.to_numpy(), predictions=train_preds)

    test['TARGET'] = test_preds
    test.to_csv('test_predicted.csv', index=False)
    print(f'pfbeta train = {pfbeta_train:.4f}, pfbeta val = {pfbeta_val:.4f}')
