import os
import re
import shutil
import xlrd
import torch
import random

attr_names = ['ID number', 'gender（0=female；1=male）', 'age', 'BMI', 
            'symptoms(0=presence,1=absence）', 'Surgery（partial=1，radical=2）',
            'pT stage', 'Furhman', 'Pathology necrosis(0,1)', 'Pathology bleeding(0,1)',
            'PFS-endpoint', 'PFS', 'OS-endpoint', 'OS',]

def read_xlsx(file_path, attr_names: list):
    data = xlrd.open_workbook(file_path)
    table = data.sheets()[0]

    attributes = []
    patient_id = []
    i = 0
    for name in table.row_values(0):
        if name in attr_names:
            if name == 'ID number':
                patient_id.append(table.col_values(i)[1:])
            else:
                attributes.append(torch.tensor(table.col_values(i)[1:]))
        i += 1

    return patient_id, torch.stack(attributes)

def read_suffixes(file_path, tgt='pfs', value=1):
    """
    Read suffixes from a text file, each suffix on a new line.

    Args:
    file_path (str): The path of the text file containing the suffixes.

    Returns:
    list: A list of suffixes.
    """
    patient_id, attributes = read_xlsx(file_path, attr_names)
    pfs = attributes[9, :] # PFS-endpoint
    os = attributes[11, :] # OS-endpoint
    suffixes = []
    if tgt == 'pfs':
        for idx, pfs_i in enumerate(pfs):
            if pfs_i == value:
                suffixes.append(patient_id[0][idx])
    else:
        for idx, os_i in enumerate(os):
            if os_i == value:
                suffixes.append(patient_id[0][idx])
    # with open(file_path, 'r') as file:
    #     suffixes = [line.strip() for line in file.readlines()]
    return suffixes

def find_files_with_suffixes(directory, suffixes):
    """
    Find files in the specified directory that end with any of the given suffixes after an underscore.

    Args:
    directory (str): The path of the directory to search.
    suffixes (list): A list of suffixes to search for.

    Returns:
    list: A list of file paths that match any of the specified suffixes.
    """
    matching_files = []
    for suffix in suffixes:
        pattern = re.compile(rf'_({suffix})$')
        for root, _, files in os.walk(directory):
            for file in files:
                match = re.search(r'_(\w+)\.npy$', file)
                match_name = match.group(1)
                if match_name == suffix:
                    matching_files.append(os.path.join(root, file))
    return matching_files

def copy_files_to_directory(files, destination_directory):
    """
    Copy specified files to the destination directory.

    Args:
    files (list): A list of file paths to be copied.
    destination_directory (str): The path of the directory to copy files to.
    """
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    
    for file in files:
        shutil.move(file, destination_directory)
        print(f"Copied {file} to {destination_directory}")

# Example usage
classes = '0'
value = int(classes)
directory_path = '../data/Jia301/preprocessed'
suffix_file_path = '../data/Jia301/attributes.xlsx'
destination_directory = f'../data/Jia301/{classes}'

# Read suffixes from the text file
suffixes = read_suffixes(suffix_file_path, value=value)

if classes=='0':
    number = 140
    new_suffixes = random.sample(suffixes, number)
    suffixes = None
    suffixes = new_suffixes

# Find matching files in the specified directory
matching_files = find_files_with_suffixes(directory_path, suffixes)

# Copy matching files to the destination directory
copy_files_to_directory(matching_files, destination_directory)

print("All matching files have been copied.")
