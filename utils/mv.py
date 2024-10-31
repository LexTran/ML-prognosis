import shutil
import os

dataset_idx = Zhao301

for src in sorted(os.listdir(f"../data/{dataset_idx}/预后一中心149")):
    if src.split(".")[-1] == "xlsx":
        continue
    if not os.path.isdir(f"../data/{dataset_idx}/预后一中心149_nii/{src}"):
        if not os.path.exists(f"../data/{dataset_idx}/核分级一中心179_nii/{src}"):
            print(src)
            continue
        shutil.copytree(f"../data/{dataset_idx}/核分级一中心179_nii/{src}", f"../data/{dataset_idx}/预后一中心149_nii/{src}")