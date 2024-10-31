import os
import argparse

import numpy as np
from scipy import stats
import pandas as pd
import xlrd

parser = argparse.ArgumentParser(description='Radiomics selecting')
parser.add_argument('-dataset_idx', '--dataset_idx', type=str, default='Zhao301', help='Which dataset to use')
args, _ = parser.parse_known_args()

args = parser.parse_args()

dataset_idx = args.dataset_idx

all_df = pd.read_csv(f'./data/{dataset_idx}/roi_radiomics.csv')
label = pd.read_excel(f'./data/{dataset_idx}/prognosis.xlsx')
label = label[['ID number', 'OS-endpoint', 'PFS-endpoint', 'OS', 'PFS']]

# move each modal radiomics to a new column
new_radioms = all_df.drop(['nii_path', 'file_id', 'patient_id'], axis=1)

radioms1 = new_radioms[new_radioms.index%4==0] # 1 modal
radioms2 = new_radioms[new_radioms.index%4==1] # 2 modal
radioms3 = new_radioms[new_radioms.index%4==2] # 3 modal
radioms4 = new_radioms[new_radioms.index%4==3] # 4 modal
radioms1.columns = [head if head=='ID number' else head+'_1' for head in radioms1.columns]
radioms2.columns = [head if head=='ID number' else head+'_2' for head in radioms2.columns]
radioms3.columns = [head if head=='ID number' else head+'_3' for head in radioms3.columns]
radioms4.columns = [head if head=='ID number' else head+'_4' for head in radioms4.columns]

# merged_norm_radioms = radioms1
merged_norm_radioms = pd.merge(radioms1, radioms2)
merged_norm_radioms = pd.merge(merged_norm_radioms, radioms3)
merged_norm_radioms = pd.merge(merged_norm_radioms, radioms4)
all_df = merged_norm_radioms
label['ID number'] = label['ID number'].astype(str)
all_df = pd.merge(all_df, label)
id_series = all_df['ID number']

significant_feat = []
for column in all_df.columns[1:]:
    W, p_shapiro = stats.shapiro(all_df[column])
    if p_shapiro <= 0.05:
        # 正态分布，使用独立样本t检验
        t, p_t = stats.ttest_ind(all_df[column], all_df['PFS-endpoint'])
        if p_t >= 0.05:
            continue
        t, p_t = stats.ttest_ind(all_df[column], all_df['OS-endpoint'])
        if p_t >= 0.05:
            continue
        t, p_t = stats.ttest_ind(all_df[column], all_df['PFS'])
        if p_t >= 0.05:
            continue
        t, p_t = stats.ttest_ind(all_df[column], all_df['OS'])
        if p_t<0.05:
            significant_feat.append(column)
    elif p_shapiro > 0.05:
        # 非正态分布，使用Mann-Whitney U检验
        W, p_mann = stats.mannwhitneyu(all_df[column], all_df['PFS-endpoint'])
        if p_mann >= 0.05:
            continue
        W, p_mann = stats.mannwhitneyu(all_df[column], all_df['OS-endpoint'])
        if p_mann >= 0.05:
            continue
        W, p_mann = stats.mannwhitneyu(all_df[column], all_df['PFS'])
        if p_mann >= 0.05:
            continue
        W, p_mann = stats.mannwhitneyu(all_df[column], all_df['OS'])
        if p_mann<0.05:
            significant_feat.append(column)

all_df = all_df[significant_feat]
# compute spearsman correlation
corr = all_df.corr(method='spearman')
# delete those columns with correlation coefficient > 0.9
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i, j] > 0.9:
            if columns[j]:
                columns[j] = False

selected_columns = all_df.columns[columns]
all_df = all_df[selected_columns]
all_df['ID number'] = id_series
all_df.to_csv(f'./data/{dataset_idx}/selectedRadioms.csv', index=False)