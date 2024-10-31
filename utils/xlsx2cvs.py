# convert xlsx files to cvs files
import pandas as pd
import xlrd

dataset_idx = 'Zhao301'

xlsx_file = pd.read_excel(f'../data/{dataset_idx}/核分级外部验证46/核分级外部验证.xlsx')
xlsx_file.to_csv(f'../data/{dataset_idx}/核分级外部验证46/核分级外部验证.csv', index=False)