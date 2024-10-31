import os
import shutil
import glob

ori_path = '../data/Zhao301/预后一中心149_nii/'
tgt_path = '../data/Zhao301/预后一中心143_nii/'

label_path = '../data/Zhao301/143ROI/'
label_files = sorted(os.listdir(label_path))

# for sub_dir in sorted(os.listdir(ori_path)):
#     if sub_dir not in label_files:
#         continue
#     shutil.copytree(ori_path + sub_dir, tgt_path + sub_dir)

for sub_dir in sorted(os.listdir(tgt_path)):
    for file in sorted(os.listdir(tgt_path + sub_dir)):
        if file.endswith('.nii.gz'):
            prefix = file.split('.nii.gz')[0]
            if prefix=='1' or prefix=='4':
                os.remove(tgt_path + sub_dir+'/'+file) 
            # shutil.copy(ori_path + sub_dir + '/' + file, tgt_path + sub_dir + '/' + file)    