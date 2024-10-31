import SimpleITK as sitk
import os
import numpy as np
import shutil
import sys
import glob
import re
import torchio.transforms as tfs 
import torch

def dcm2nii(par_dcm, nii_dir):
    if not os.path.exists(nii_dir):
        os.makedirs(nii_dir)
    for sub_dir in os.listdir(par_dcm):
        if sub_dir.endswith('.xlsx'):
            continue
        if not os.path.isdir(os.path.join(nii_dir, sub_dir)):
            os.makedirs(os.path.join(nii_dir, sub_dir))
        for dcm_dir in os.listdir(par_dcm + '/' + sub_dir):
            name = dcm_dir
            if os.path.exists(os.path.join(nii_dir+'/'+sub_dir, f'{name}.nii.gz')):
                continue
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(par_dcm+'/'+sub_dir+'/'+dcm_dir)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            ori_spacing = image.GetSpacing()
            ori_size = image.GetSize()
            image = preprocess(image)
            image = sitk.GetImageFromArray(image.squeeze(0).numpy())
            image.SetSpacing(ori_spacing)
            image.SetOrigin(image.GetOrigin())
            image.SetDirection(image.GetDirection())
            sitk.WriteImage(image, os.path.join(nii_dir+'/'+sub_dir, f'{name}.nii.gz'))

def preprocess(dcm_img):
    # N4 bias correction
    dcm_img = sitk.Cast(dcm_img, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    dcm_img = corrector.Execute(dcm_img)
    # Normalize
    dcm_array = sitk.GetArrayFromImage(dcm_img).astype(np.float32)
    dcm_tensor = torch.Tensor(dcm_array).unsqueeze(0) # Convert to tensor
    dcm_tensor = tfs.ZNormalization()(dcm_tensor) # normalize
    return dcm_tensor


if __name__ == '__main__':
    dcm2nii(sys.argv[1], sys.argv[2])
    print('done')