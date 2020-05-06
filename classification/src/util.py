import os
import pandas as pd

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

def get_size(matrix):
    size = matrix.data.nbytes + matrix.indptr.nbytes + matrix.indices.nbytes
    size = size / 1024 / 1024
    print("size = {:.2f} MB".format(size))


