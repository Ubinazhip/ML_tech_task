# ML_tech_task
# Feature Selection(feature_selection.py)
- Remove columns(features) where 40% of entries are None. 
- Remove constant columns
- Remove quasi constant columns(almost constant - 99% of data is one number)
- Remove duplicated columns
- Remove correlated columns, if correlation >= 0.8
-After feature selection we have only 118 features
# Preprocessing(Preprocessing.py)
- FillNa numerical columns with with their mean
- FillNa object columns with most_frequent strategy
- One hot encoding of object columns
- Standard Scaler of numerical columns - (x - mean)/std
- train/val/test sets after preprocessing is [here](https://drive.google.com/drive/folders/1I_9fAZyag02Gy9SdFNVCelLDbyCMgaqO?usp=sharing)
# Target metric
- [probabilistic F1 score](https://aclanthology.org/2020.eval4nlp-1.9.pdf) - works good for imbalance dataset
# Training
- XGBoostClassifier
- Best parameter search through GridSearchCV
# Test
- Test Results are saved in test_predicted.csv, in column 'TARGET'
# runner.py
- Run runner.py for feature selection, preprocessing and training with best parameters found using GridSearchCV
# Future work
- Dimensionality reduction techniques like PCA
- Deal with class imbalance
- Better feature selection, preprocessing
- Better gridSearch for finding best parameters - I did it only once because of lack of tine
 
