import os
import six
import glob
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from radiomics import featureextractor

parser = argparse.ArgumentParser(description='Calculate radiomics')
parser.add_argument('-dataset_idx', '--dataset_idx', type=str, default='Zhao301', help='Which dataset to use')
parser.add_argument('-config', '--rad_config', type=str, default='./config/params.yaml', help='Configuration for pyRadioms')
args, _ = parser.parse_known_args()

args = parser.parse_args()

# Define radiomics
extractor = featureextractor.RadiomicsFeatureExtractor(args.rad_config)
image_path = Path(f'./data/{dataset_idx}/预后一中心143_nii/')
label_path = Path(f'./data/{dataset_idx}/143ROI/')

def calc_radiomics(image, mask):
    # Normalize physics information
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)
    image = sitk.GetImageFromArray(image_array)
    mask = sitk.GetImageFromArray(mask_array)
    # mask = sitk.GetImageFromArray((image_array>0).astype(np.uint8))
    # mask.CopyInformation(image)
    result = extractor.execute(image, mask)
    return result

# resample
def resample(oriImage, tgtSize):
    euler = sitk.Euler3DTransform()
    if oriImage.GetDimension() == 4:
        x, y, z, _ = oriImage.GetSize()
    elif oriImage.GetDimension() == 3:
        x, y, z = oriImage.GetSize()
    x_spacing, y_spacing, z_spacing = oriImage.GetSpacing()
    width, height, depth = tgtSize
    new_spacing_x, new_spacing_y, new_spacing_z = x_spacing, y_spacing, z_spacing
    new_x, new_y, new_z = x, y, z
    if x != width:
        new_spacing_x = x_spacing * x / width
        new_x = round(x * x_spacing / new_spacing_x)
    if y != height:
        new_spacing_y = y_spacing * y / height
        new_y = round(y * y_spacing / new_spacing_y)
    if z != depth:
        new_spacing_z = z_spacing * z / depth
        new_z = round(z * z_spacing / new_spacing_z)

    origin = oriImage.GetOrigin()
    direction = oriImage.GetDirection()
    newsize = (new_x, new_y, new_z)
    newspce = (new_spacing_x, new_spacing_y, new_spacing_z)
    newimage = sitk.Resample(oriImage, newsize, euler, sitk.sitkLinear, origin, newspce, direction, 0.0, sitk.sitkFloat32)
    new_array = sitk.GetArrayFromImage(newimage)
    new_array[new_array >= 0.5] = 1
    new_array[new_array < 0.5] = 0
    returnImage = sitk.GetImageFromArray(new_array)
    returnImage.CopyInformation(newimage)
    return returnImage

# define result table
if os.path.isfile(f'./data/{dataset_idx}/roi_radiomics.csv'):
    all_df = pd.read_csv(f'./data/{dataset_idx}/roi_radiomics.csv')
else:
    all_df = pd.DataFrame({'nii_path': list(image_path.glob('*/*.nii.gz'))})
    all_df['file_id'] = all_df['nii_path'].map(lambda x: x.stem)
    all_df['patient_id'] = all_df['nii_path'].map(lambda x: x.parent.stem)
    all_df['ID number'] = all_df['nii_path'].map(lambda x: str(x.parent).split('_')[-1])

# feature extraction
mask_names = {'0': 0, '2': 1, '3': 2, '5': 3}
for sub_dir in sorted(os.listdir(image_path)):
    label_list = sorted(glob.glob(str(label_path / sub_dir / '*.nrrd')))
    if label_list == []:
        label_list = sorted(glob.glob(str(label_path / sub_dir / '*.nii.gz')))
    for file in sorted(os.listdir(image_path / sub_dir)):
        if file.endswith('.nii.gz'):
            imageName = file
        # if('original_firstorder_Mean' not in all_df.columns):
        if all_df.loc[(all_df['patient_id']==sub_dir)&(all_df['file_id']==imageName.split('.gz')[0])]['original_firstorder_Mean'].isna().all():
            #     continue
            try:
                prefix = imageName.split('.nii.gz')[0]
                # if this row not in table, append new row
                if ~all_df.empty & all_df.loc[(all_df['file_id']==prefix+'.nii')&(all_df['patient_id']==sub_dir)].empty:
                    pre_prefix = mask_names[prefix]-1
                    pre_prefix = [k for k,v in mask_names.items() if v == pre_prefix][0]
                    index = all_df.loc[(all_df['file_id']==pre_prefix+'.nii')&(all_df['patient_id']==sub_dir)].index.values[0]
                    pre_df = all_df[:index+1]
                    post_df = all_df[index+1:]
                    all_df = pd.concat([pre_df, pd.DataFrame({'nii_path': all_df.iloc[index]['nii_path'], 'scan_id': all_df.iloc[index]['scan_id'],'file_id': prefix+'.nii', 'patient_id': sub_dir}, index=[0])], ignore_index=True)
                    all_df = pd.concat([all_df, post_df], ignore_index=True)
                    
                if prefix in mask_names:
                    for label in label_list:
                        if prefix in label.split('/')[-1].split('_mask')[0] \
                            or prefix in label.split('/')[-1].split('-labels')[0]\
                            or prefix in label.split('/')[-1].split('_')[-1]:
                            mask = label
                            label_list.remove(label)
                            break
                    # mask = [label_list[i] for i in range(len(label_list)) if prefix in label_list[i].split('/')[-1]]
                else:
                    # all_df.to_csv('roi_radiomics_table.csv', index=False)
                    # continue
                    print(f'{sub_dir} {file} not in mask_names')
                    break
                
                # if 'CUI DA BI_C159481' in mask[0]:
                print(f'Processing {sub_dir} {file}')
                print(f'Mask {mask}')
                image = sitk.ReadImage(image_path / sub_dir / imageName)
                mask = sitk.ReadImage(mask)
                # if(image.GetSize() == mask.GetSize()):
                #     continue
                if mask.GetDimension() == 4:
                    mask_array = sitk.GetArrayFromImage(mask)
                    # mask_array = np.squeeze(mask_array, axis=0)
                    mask_array = mask_array[0]
                    mask = sitk.GetImageFromArray(mask_array)
                    # sitk.WriteImage(mask, str(f'./re_roi/{sub_dir}_{prefix}.seg.nii.gz'))
                if image.GetSize() != mask.GetSize():
                    if os.path.isfile(f'./data/{dataset_idx}/re_roi/{sub_dir}_{prefix}.seg.nii.gz'):
                        mask = sitk.ReadImage(f'./data/{dataset_idx}/re_roi/{sub_dir}_{prefix}.seg.nii.gz')
                    else:
                        print(f'{sub_dir} image and mask size not match')
                        new_mask = resample(mask, image.GetSize())
                        sitk.WriteImage(new_mask, str(f'../data/{dataset_idx}/re_roi/{sub_dir}_{prefix}.seg.nii.gz'))
                        mask = new_mask
                    # continue

                # calculate radiomics
                result = calc_radiomics(image, mask)
                for key, val in six.iteritems(result):
                    if 'diagnostics' in key:
                        continue
                    all_df.loc[(all_df['file_id']==imageName.split('.gz')[0])&(all_df['patient_id']==sub_dir), key] = val

            except Exception as e:
                print(sub_dir, file + ' error')
                all_df.to_csv(f'./data/{dataset_idx}/roi_radiomics.csv', index=False)
                continue

# export the whole table
all_df.to_csv(f'./data/{dataset_idx}/roi_radiomics.csv', index=False)
print('Done!')