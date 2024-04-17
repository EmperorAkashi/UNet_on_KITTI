import os
import numpy as np
from typing import Optional
import collections

def read_from_folder(path: str)->dict:
    """
    Creates the data dictionary directly from the folder without a .json or yaml file. Only suitable for simple datasets.
    Folders should have the same name as keys. Folder structure is assumed to be as follows:
    <path>
    color
        <image_01>
        ...
        <image_n>
    semantic
        <image_01>
        ...
        <image_n>
    ...
    :param path: path of the dataset/dataset split to use
    :return: a dictionary with all files for each key, sorted alphabetically by filename
    """
    sub_folder = list_subdir(path) # path in dataconfig should be the top folder
    semantic_dataframe = collections.defaultdict()

    for f in sub_folder:
        if f[0] == '.':
            continue
        folder_name = split_path(f)
        semantic_dataframe[folder_name] = list_files_in_dir(f)
    return semantic_dataframe


def list_subdir(curr_dir: str) -> list:
    """ method that lists all subdirectories of a given directory
        :param 
            curr_dir: directory in which the subdirectories are searched in
    """
    top_dir = os.path.abspath(curr_dir)
    sub_dir = [os.path.join(top_dir, i) for i in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, i)) or i[0] != '.']
    return sub_dir

def list_files_in_dir(top_dir: str) -> list:
    """ method that lists all files of a given directory
    :param 
        top_dir: directory in which the files are searched in
    """
    top_dir = os.path.abspath(top_dir)
    files = [os.path.join(top_dir, x) for x in os.listdir(top_dir)
                if os.path.isfile(os.path.join(top_dir, x))]
    return files

def split_path(path: str) -> str:
    list_folder = path.split('/')
    return list_folder[-1]