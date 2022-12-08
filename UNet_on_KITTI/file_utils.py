import os
import numpy as np
from typing import Optional

def read_from_folder(path: str):
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

def list_files_in_dir(curr_dir: str):
    """ method that lists all subdirectories of a given directory
        :param top_dir: directory in which the subdirectories are searched in
    """
    top_dir = os.path.abspath(curr_dir)
    sub_dir = [os.path.join(top_dir, i) for i in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, i)) or i[0] != '.']
    return sub_dir

def list_files_in_dir(top_dir):
    """ method that lists all files of a given directory
    :param top_dir: directory in which the files are searched in
    """
    top_dir = os.path.abspath(top_dir)
    files = [os.path.join(top_dir, x) for x in os.listdir(top_dir)
                if os.path.isfile(os.path.join(top_dir, x))]
    return files