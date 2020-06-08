import os
import pandas as pd
import h5py
import numpy as np
from config import *

class EarlyStop:
    def __init__(self, mode="max", history=5):
        if mode == "max":
            self.best_func = np.max
            self.best_val = -np.inf 
            self.comp = lambda x, y: x >= y
        elif mode == "min":
            self.best_func = np.min
            self.best_val = np.inf 
            self.comp = lambda x, y: x <= y
        else:
            print("Please use 'max' or 'min' for mode.")
            quit()
        
        self.history_num = history
        self.history = np.zeros((self.history_num, ))
        self.total_num = 0

    def check(self, score):
        self.history[self.total_num % self.history_num] = score
        self.total_num += 1
        current_best_val = self.best_func(self.history)
        
        if self.total_num <= self.history_num:
            return False

        if self.comp(current_best_val, self.best_val):
            self.best_val = current_best_val
            return False
        else:
            return True

def create_dir(path):
    try:
        os.mkdir(path)
    except Exception as e:
        print(e)

def output_score(score):
    table = pd.DataFrame(
        [score[3], score[0], score[1], score[2]],
        index=["# samples", "Precision", "Recall", "F1"],
        columns=["background", "purpose", "method", "finding", "other"],
    )
    return table

def output_confusion(confusion):
    table = pd.DataFrame(
        confusion,
        index=["background", "purpose", "method", "finding", "other"],
        columns=["background", "purpose", "method", "finding", "other"],
    )
    return table

def get_size(matrix):
    size = matrix.data.nbytes + matrix.indptr.nbytes + matrix.indices.nbytes
    size = size / 1024 / 1024
    print("size = {:.2f} MB".format(size))

def h5_save(x, y, filename):
    with h5py.File(filename, 'w') as outfile:
        outfile.create_dataset("x", data=y) 
        outfile.create_dataset("y", data=y)

def h5_load(filename, data_list, dtype=None, verbose=False):
    with h5py.File(filename, 'r') as infile:
        data = []
        for data_name in data_list:
            if dtype is not None:
                temp = np.empty(infile[data_name].shape, dtype=dtype)
            else:
                temp = np.empty(infile[data_name].shape, dtype=infile[data_name].dtype)
            infile[data_name].read_direct(temp)
            data.append(temp)
          
        if verbose:
            print("\n".join(
                "{} = {} [{}]".format(data_name, str(real_data.shape), str(real_data.dtype))
                for data_name, real_data in zip(data_list, data)
            ))
            print()
        return data

