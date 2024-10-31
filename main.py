import random
import os
import glob
import math
import argparse

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import silhouette_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
from xgboost import XGBClassifier
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='ML Radiomics Prognosis')
parser.add_argument('-dataset_idx', '--dataset_idx', type=str, default='Zhao301', help='Which dataset to use')
parser.add_argument('-radioms', '--isRadiomsUsed', type=bool, default=False, help='Whether to use radiomics or not')
parser.add_argument('-pca', '--isPCAUsed', type=bool, default=False, help='Whether to use PCA to decrease dimension')
parser.add_argument('-tgt', '--tgt', type=str, default='PFS', help='Which metric to prognosis')
parser.add_argument('-output', '--out_dir', type=str, default=None, help='Where to output')
args, _ = parser.parse_known_args()

args = parser.parse_args()


def visualize_result(y_pred, y_test, prob, title): 
    '''
        Function to plot confusion matrix and ROC curve 
        
        Args: 
            y_pred: predicted values
            y_test: actual values
            title: title of the plot
            
        Example Usage: 
            plot_clf(y_pred, y_test, 'Random Forest')
    '''
    print(f"Accuracy_{title}: ", accuracy_score(y_test, y_pred))
    print(f"AUC_{title}: ", roc_auc_score(y_test, prob))

    # Create the output directory if not exists
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = './results/' + args.dataset_idx
        if args.isRadiomsUsed:
            out_dir += '_rad'
        if args.isPCAUsed:
            out_dir += '_pca'
        out_dir += f'_{args.tgt}'
        if os.path.isdir(out_dir) is not True:
            os.makedirs(out_dir)

    fpr, tpr, _ = metrics.roc_curve(y_test, prob)
    roc_auc = metrics.auc(fpr, tpr)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=title)
    roc_display.plot()
    plt.savefig(f'{out_dir}/roc_{title}.png')

    conf_mat = confusion_matrix(y_test, y_pred)
    conf_display = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=np.array([0, 1]))
    conf_display.plot()
    plt.savefig(f'{out_dir}/confusion_matrix_{title}.png')

# Read data
dataset_idx  = args.dataset_idx
if args.dataset_idx == 'Jia301':
    attr_names = ['ID number', 'gender（0=female；1=male）', 'age', 'BMI', 
                'symptoms(0=presence,1=absence）', 'Surgery（partial=1，radical=2）',
                'pT stage', 'Furhman', 'Pathology necrosis(0,1)', 'Pathology bleeding(0,1)',
                'PFS-endpoint', 'PFS', 'OS-endpoint', 'OS',]
    onehot_names = ['gender（0=female；1=male）', 'symptoms(0=presence,1=absence）',
                'Surgery（partial=1，radical=2）', 'Furhman',
                'Pathology necrosis(0,1)', 'Pathology bleeding(0,1)']
    data = pd.read_excel(f'./data/{dataset_idx}/prognosis.xlsx', sheet_name='MRI筛选2')

elif args.dataset_idx == 'Zhao301':
    attr_names = ['ID number', 'gender（0=female；1=male）', 'age', 'BMI', 
                'symptoms(0=absence,1=presence）', 'Surgery（partial=1，radical=2）',
                'pT stage', 'nuclear grade', 'Pathology necrosis(0,1)', 'Pathology bleeding(0,1)',
                'PFS-endpoint', 'PFS', 'OS-endpoint', 'OS',]
    onehot_names = ['gender（0=female；1=male）', 'symptoms(0=absence,1=presence）',
                'Surgery（partial=1，radical=2）', 'nuclear grade',
                'Pathology necrosis(0,1)', 'Pathology bleeding(0,1)']
    data = pd.read_excel(f'./data/{dataset_idx}/prognosis.xlsx', sheet_name='Sheet1')

scale_names = ['age', 'BMI']

radioms = None
radioms = pd.read_csv(f'./data/{dataset_idx}/selectedRadioms.csv')

for name in data.columns:
    if name not in attr_names:
        data = data.drop(name, axis=1)

col_cov = data.columns # column names, feature names
row_cov = data.index # row names, patient ID

data['ID number'] = data['ID number'].astype(str)
if args.isRadiomsUsed:
    X = pd.merge(data, radioms, left_on='ID number', right_on='ID number')
else:
    X = data
y_pfs_end = X['PFS-endpoint']
y_os_end = X['OS-endpoint']
y_pfs = X['PFS']
y_os = X['OS']
X = X.drop(['PFS-endpoint', 'OS-endpoint', 'PFS', 'OS'], axis=1)

X.to_csv(f'./data/{dataset_idx}/merged.csv', index=False)
X = X.drop(['ID number'], axis=1)

if dataset_idx == 'Jia301':
    X = X.drop(['pT stage'], axis=1)
elif dataset_idx == 'Zhao301':
    X.loc[X['pT stage'] == '1a', 'pT stage'] = 1
    X.loc[X['pT stage'] == '1b', 'pT stage'] = 1
    X.loc[X['pT stage'] == '2a', 'pT stage'] = 2
    X.loc[X['pT stage'] == '2b', 'pT stage'] = 2
    X.loc[X['pT stage'] == '3a', 'pT stage'] = 3
    X.loc[X['pT stage'] == '3b', 'pT stage'] = 3
    X.loc[X['pT stage'] == '3c', 'pT stage'] = 3

    X.loc[X['nuclear grade'] == '1', 'nuclear grade'] = 1
    X.loc[X['nuclear grade'] == '2', 'nuclear grade'] = 2
    X.loc[X['nuclear grade'] == '3', 'nuclear grade'] = 3
    X.loc[X['nuclear grade'] == 'x', 'nuclear grade'] = 4

# PCA select
if args.isPCAUsed:
    pca = PCA(n_components=0.9)
    pca_feat = X.iloc[:, 9:]
    pca_feat = pca.fit_transform(pca_feat)
    X = X.drop(X.columns[9:], axis=1)
    pca_feat = pd.DataFrame(pca_feat)
    pca_feat.columns = [f'pca_{str(i)}' for i in range(pca_feat.shape[1])]
    X = pd.concat([X, pca_feat], axis=1)

if args.tgt == 'PFS':
    X_train, X_test, y_train, y_test = train_test_split(X, y_pfs_end, test_size=0.2, random_state=7)
elif args.tgt == 'OS':
    X_train, X_test, y_train, y_test = train_test_split(X, y_os_end, test_size=0.2, random_state=7)
# X_train, X_test, y_train, y_test = train_test_split(X, y_pfs, test_size=0.2, random_state=7)
# X_train, X_test, y_train, y_test = train_test_split(X, y_os, test_size=0.2, random_state=7)
scaler = StandardScaler()
ct = ColumnTransformer([
    ('scaler', scaler, scale_names),
    ('one-hot', OneHotEncoder(sparse_output=False), onehot_names)
    ], remainder='passthrough'
)
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

# Logistic Regression

# param_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],
#     'penalty': ['l1', 'l2', 'elasticnet'],
#     'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
#     'max_iter': [100, 500, 1000]
# }
# model = LogisticRegression()
# grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
# grid.fit(X_train, y_train.values.ravel())
# print("Best Parameters for Logistic:", grid.best_params_)
# print("Best ROC_AUC Score for Logistic:", grid.best_score_)

model = LogisticRegression(C=1, max_iter=100, penalty='l1', class_weight='balanced', solver='liblinear')
model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
visualize_result(y_pred, y_test, y_prob, 'Logistic Regression')

# Random Forest

# param_grid = {
#     'n_estimators': [40, 80, 160, 320, 640, 1280],
#     'min_samples_split': [8, 10, 12, 24],
#     'max_depth': [2, 4, 8]
# }
# model = RandomForestClassifier()
# grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
# grid.fit(X_train, y_train.values.ravel())
# print("Best Parameters for Random Forest:", grid.best_params_)
# print("Best ROC_AUC Score for Random Forest:", grid.best_score_)

model = RandomForestClassifier(n_estimators=80, min_samples_split=24, max_depth=8)
model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
visualize_result(y_pred, y_test, y_prob, 'Random Forest')


# xgboost

# param_grid = {
#     'tree_method': ['exact'],
#     'n_estimators': [20, 40, 80, 100, 500, 1000],
#     'max_depth': range(3,10,2),
#     'min_child_weight': range(1,6,2),
#     'gamma': [i/10.0 for i in range(0,5)],
#     'subsample': [0.73, 0.8, 0.85, 0.9, 0.95],
#     'reg_alpha': [1e-2, 0.1, 1, 10],
#     'learning_rate': [0.0005, 0.001, 0.01, 0.05],
#     'scale_pos_weight': [1, 5, 8, 10, 50],
# }
# model = XGBClassifier()
# grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
# grid.fit(X_train, y_train.values.ravel())
# print("Best Parameters for xgboost:", grid.best_params_)
# print("Best ROC_AUC Score for xgboost:", grid.best_score_)

model = XGBClassifier(
    learning_rate=0.05,
    n_estimators=40,         # 树的个数--1000棵树建立xgboost
    max_depth=15,               # 树的深度
    min_child_weight = 10,      # 叶子节点最小权重
    gamma=0.4,                # 惩罚项中叶子结点个数前的参数
    # gamma=1.2,                  # 惩罚项中叶子结点个数前的参数
    subsample=0.8,             # 随机选择80%样本建立决策树
    colsample_btree=0.8,       # 随机选择80%特征建立决策树
    # objective='multi:softmax', # 指定损失函数
    # scale_pos_weight=4,        # 解决样本个数不平衡的问题
    # scale_pos_weight=3,        # 解决样本个数不平衡的问题
    scale_pos_weight=5,        # 解决样本个数不平衡的问题
    random_state=27,           # 随机数
    # num_class=2,
    reg_alpha=10,
    reg_lambda=15,
)
model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
visualize_result(y_pred, y_test, y_prob, 'xgboost')