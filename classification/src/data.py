import json
import os
from config import *
import numpy as np

def load_data(phrase="train", verbose=False):
    filenames = [f for f in os.listdir(os.path.join(covid_data_dir, phrase)) if ".swp" not in f and ".json" in f]
    if verbose:
        print("Loading {} data".format(phrase))
        print("# file", len(filenames))

    x = []
    y = []
    for filename in filenames:
        with open(os.path.join(covid_data_dir, phrase, filename), 'r', encoding='utf-8') as infile:
            data = json.load(infile)
            x.extend([
                segment["segment_text"]
                for paragraph in data["abstract"]
                for sent in paragraph["sentences"]
                for segment in sent
            ])
            y.extend([
                segment["crowd_label"]
                for paragraph in data["abstract"]
                for sent in paragraph["sentences"]
                for segment in sent
            ])
    
    if verbose:
        print("# x", len(x))
        print("# y", len(y))

        # stat for x
        x_stat = np.array([len(xx.split()) for xx in x])
        print("avg seq length = {:.3f} (SD={:.3f})".format(x_stat.mean(), x_stat.std()))
        print("min = {}, max = {}".format(x_stat.min(), x_stat.max()))

        print()

    return x, y

def test():
    x_train, y_train = load_data("train", verbose=True)
    x_test, y_test = load_data("test", verbose=True)
    x_dev, y_dev = load_data("dev", verbose=True)


if __name__ == "__main__":
    test()
