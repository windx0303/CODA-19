from data import load_data
from config import *
from util import *

import joblib
import os
import numpy as np
import pandas as pd
import argparse
from collections import Counter
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils import data

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.ensemble import RandomForestClassifier

RAND_SEED = 15234

#########################################################
# ML Baseline
#########################################################
def preprocess_ml_data(version):
    ml_model_dir = os.path.join(model_dir, version)

    # load data
    x_train, y_train = load_data(phrase="train", verbose=True)
    x_test, y_test = load_data(phrase="test", verbose=True)
    x_dev, y_dev = load_data(phrase="dev", verbose=True)

    # build tfidf vectorization model
    tfidf_model = TfidfVectorizer(lowercase=True, min_df=5)
    tfidf_model.fit(x_train)
    with open(os.path.join(ml_model_dir, "tfidf.model"), 'wb') as outfile:
        joblib.dump(tfidf_model, outfile)

    x_train = tfidf_model.transform(x_train)
    x_test = tfidf_model.transform(x_test)
    x_dev = tfidf_model.transform(x_dev)

    print("x_train", x_train.shape)
    print("x_test", x_test.shape)
    print("x_dev", x_dev.shape)
    
    # turn label into vector
    y_train = np.array([label_mapping[y] for y in y_train])
    y_test = np.array([label_mapping[y] for y in y_test])
    y_dev = np.array([label_mapping[y] for y in y_dev])

    print("y_train", y_train.shape)
    print("y_test", y_test.shape)
    print("y_dev", y_dev.shape)
    print()

    return (x_train, y_train), (x_test, y_test), (x_dev, y_dev)

def ml_baseline(arg):
    from ml_config import parameter_dict

    version = arg.model
    ml_model_dir = os.path.join(model_dir, version)
    create_dir(ml_model_dir)
    print("Training {} model\n".format(version))

    # load data
    (x_train, y_train), (x_test, y_test), (x_dev, y_dev) = preprocess_ml_data(version)
    parameter_list = parameter_dict[version]

    # parameter search
    best_model = None
    best_parameter = None
    best_score = 0.0
    for model_index, parameter in enumerate(parameter_list, 1):
        print("Running {}-th model (total = {})".format(model_index, len(parameter_list)))
        print(str(parameter))

        if version == "SVM":
            model = LinearSVC(verbose=False, random_state=RAND_SEED, **parameter)
        elif version == "RandomForest":
            model = RandomForestClassifier(n_jobs=5, random_state=RAND_SEED, **parameter)
        model.fit(x_train, y_train)

        # test on dev
        y_dev_pred = model.predict(x_dev)
        acc = accuracy_score(y_dev, y_dev_pred)

        # check best
        if acc > best_score:
            best_model = model
            best_parameter = parameter
            best_score = acc

        # log
        print("Dep Acc = {}\n".format(acc))
        score = precision_recall_fscore_support(y_dev, y_dev_pred)
        table = output_score(score)
        with open(os.path.join(log_dir, "{}.log".format(version)), 'a', encoding='utf-8') as outfile:
            outfile.write(str(parameter)+"\n")
            outfile.write(str(score)+"\n")
            outfile.write("acc = {}\n\n".format(acc))

    # test with the best model
    y_test_pred = best_model.predict(x_test)
    score = precision_recall_fscore_support(y_test, y_test_pred)
    table = output_score(score)
    print(best_parameter)
    print(table)

    acc = accuracy_score(y_test, y_test_pred)
    print("Accuracy", acc)

    with open(os.path.join(result_dir, "{}.result".format(version)), 'w', encoding='utf-8') as outfile:
        outfile.write(str(best_parameter)+"\n")
        outfile.write(table.to_csv(path_or_buf=None)+"\n")
        outfile.write("acc = {}\n".format(acc))

    # save model
    with open(os.path.join(ml_model_dir, "model"), 'wb') as outfile:
        joblib.dump(best_model, outfile)

#########################################################
# DL Baseline
#########################################################
class CovidDataset(data.Dataset):
    def __init__(self, x, y, bucket_num=None):
        self.x = x.astype(np.int64)
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.y.shape[0] 

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class Dictionary:
    def __init__(self, lowercase=False, min_count=5, pad_length=60):
        self.lowercase = lowercase
        self.min_count = min_count
        self.pad_length = pad_length
        self.dictionary = {
            "<pad>": 0,
            "<unk>": 1,
        }

    def get_vocab_size(self):
        return len(self.dictionary)

    def fit(self, documents):
        # count freq
        counter = Counter()
        for doc in documents:
            if self.lowercase:
                doc = doc.lower()

            tokens = doc.split(" ")
            counter.update(tokens)

        # remove tokens with 
        # TODO: freq > or >= min_count
        for token, freq in counter.items():
            if freq > self.min_count:
                self.dictionary[token] = len(self.dictionary)

    def transform(self, documents):
        if len(self.dictionary) == 2:
            print("Please fit the dictionary first.")
            quit()

        if self.lowercase:
            documents = (doc.lower() for doc in documents)

        # turn to index
        results = [
            [self.dictionary.get(token, 1) for token in doc.split(" ")]
            for doc in documents        
        ]

        # turn to matrix with padding
        matrix = np.zeros([len(results), self.pad_length], dtype=np.int32)
        for i, res in enumerate(results):
            length = min(len(res), self.pad_length)
            matrix[i, self.pad_length-length:] = res[:length]

        return matrix

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, layer_num=2, dropout_rate=0.3):
        super(LSTMClassifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, layer_num, batch_first=True, dropout=dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        vectors = self.embedding(x)
        output, _ = self.lstm(vectors)
        output = output[:, -1, :].squeeze()
        output = self.linear(output)

        return output

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, filter_list, dropout_rate=0.3):
        super(CNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        feature_num = sum(out_channel for kernel, out_channel in filter_list)
        self.linear = nn.Linear(feature_num, output_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.cnn_list = nn.ModuleList([
            nn.Conv1d(hidden_size, out_channel, kernel)
            for kernel, out_channel in filter_list
        ])
        self.pool_list = nn.ModuleList([
            nn.MaxPool1d(61-kernel)
            for kernel, out_channel in filter_list    
        ])

    def forward(self, x):
        vectors = self.embedding(x)
        vectors = vectors.transpose(1, 2)
        vectors = self.dropout(vectors)

        res = []
        for cnn, pool in zip(self.cnn_list, self.pool_list):
            output = cnn(vectors)
            output = pool(output)
            output = output.view(vectors.shape[0], -1)
            res.append(output)
        
        output = torch.cat(res, dim=1)
        output = self.linear(output)
        return output

def dl_baseline(arg):
    from dl_config import dl_parameter_dict

    # parameter setting
    parameters = dl_parameter_dict[arg.model]
    batch_size      = parameters["batch_size"]
    epoch_num       = parameters["epoch_num"]
    hidden_size     = parameters["hidden_size"]
    dropout_rate    = parameters["dropout_rate"]
    learning_rate   = parameters["learning_rate"]
    device          = parameters["device"]

    # load data
    version = arg.model
    dl_model_dir = os.path.join(model_dir, version)
    create_dir(dl_model_dir)

    # load data
    x_train, y_train = load_data(phrase="train", verbose=True)
    x_test, y_test = load_data(phrase="test", verbose=True)
    x_dev, y_dev = load_data(phrase="dev", verbose=True)

    text_dictionary = Dictionary(
        lowercase=True, 
        min_count=5
    )
    text_dictionary.fit(x_train)
    with open(os.path.join(dl_model_dir, "dictionary.model"), 'wb') as outfile:
        joblib.dump(text_dictionary, outfile)

    x_train = text_dictionary.transform(x_train)
    x_test = text_dictionary.transform(x_test)
    x_dev = text_dictionary.transform(x_dev)

    # turn label into vector
    y_train = np.array([label_mapping[y] for y in y_train])
    y_test = np.array([label_mapping[y] for y in y_test])
    y_dev = np.array([label_mapping[y] for y in y_dev])

    # build dataset
    train_dataset = CovidDataset(x_train, y_train)
    test_dataset = CovidDataset(x_test, y_test)
    dev_dataset = CovidDataset(x_dev, y_dev)
    training = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    testing = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    dev = data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("vocab size", text_dictionary.get_vocab_size())
    if arg.model == "LSTM":
        model = LSTMClassifier(
            text_dictionary.get_vocab_size(),
            hidden_size=hidden_size,
            output_size=5,
            layer_num=parameters["layer_num"],
            dropout_rate=dropout_rate,
        ).to(device)
    elif arg.model == "CNN":
        model = CNNClassifier(
            text_dictionary.get_vocab_size(),
            hidden_size=hidden_size,
            output_size=5,
            filter_list=parameters["filter_list"],
            dropout_rate=dropout_rate,
        ).to(device)
    else:
        print("Please use LSTM / CNN as model type.")
        quit()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=parameters["reg_weight"])

    evaluate(model, dev, device=device)
    best_model = copy.deepcopy(model.state_dict())
    best_accuracy = 0
    best_epoch = 0
    for epoch in range(1, epoch_num+1):
        model.train()
        total_loss = 0
        total_acc = 0
        total_count = len(train_dataset) // batch_size
        for count, (x_batch, y_batch) in enumerate(training, 1):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_batch)
    
            # compute loss
            loss = F.cross_entropy(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # compute accuracy
            y_pred = torch.argmax(y_pred, dim=1)
            correct_num = torch.sum(y_pred == y_batch).double()
            total_acc += correct_num / y_pred.shape[0]

            print("\x1b[2K\rEpoch: {} / {} [{:.2f}%] Loss: {:.5f} Acc: {:.5f}".format(
                epoch, epoch_num, 100.0*count/total_count, total_loss/count, total_acc/count), end="")

        print()
        if epoch % 1 == 0:
            acc, _, _ = evaluate(model, dev, device=device)

            if acc > best_accuracy:
                best_model = copy.deepcopy(model.state_dict())
                best_accuracy = acc
                best_epoch = epoch

    # load best model & test & save
    print("loading model from epoch {}".format(best_epoch))
    torch.save(best_model, os.path.join(dl_model_dir, "best_model.pt"))
    model.load_state_dict(best_model)
    acc, predict, true_label = evaluate(model, testing, device=device) 
    score = precision_recall_fscore_support(true_label, predict)
    table = output_score(score)
    print(table)

    # output result
    with open(os.path.join(result_dir, "{}.result".format(version)), 'w', encoding='utf-8') as outfile:
        outfile.write(table.to_csv(path_or_buf=None)+"\n")
        outfile.write("acc = {}\n".format(acc))

def evaluate(model, eval_data, device):
    model.eval()
    total_count = len(eval_data.dataset) // eval_data.batch_size
    total_loss = 0
    total_acc = 0
    predict = []
    true_label = []
    for count, (x_batch, y_batch) in enumerate(eval_data, 1):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(x_batch)

        # compute loss
        loss = F.cross_entropy(y_pred, y_batch)
        total_loss += loss.item()

        # compute accuracy
        y_pred = torch.argmax(y_pred, dim=1)
        correct_num = torch.sum(y_pred == y_batch).double()
        total_acc += correct_num / y_pred.shape[0]
        predict.append(y_pred.cpu().numpy())
        true_label.append(y_batch.cpu().numpy())

        print("\x1b[2K\rEval [{:.2f}%] Loss: {:.5f} Acc: {:.5f}".format(
            100.0*count/total_count, total_loss/count, total_acc/count), end="")
    
    predict = np.hstack(predict)
    true_label = np.hstack(true_label)
    acc = accuracy_score(true_label, predict)
    print("\nAccuracy: {:.5f}\n".format(acc))

    return acc, predict, true_label

def check_length():
    # load data
    x_train, y_train = load_data(phrase="train", verbose=True)
    x_test, y_test = load_data(phrase="test", verbose=True)
    x_dev, y_dev = load_data(phrase="dev", verbose=True)

    x_train = [x.split(" ") for x in x_train]
    x_test = [x.split(" ") for x in x_test]
    x_dev = [x.split(" ") for x in x_dev]

    # check length
    for length in [50, 60, 70, 100, 150]:
        print()
        print(length)
        print("train", sum(1 for x in x_train if len(x)>=length))
        print("test", sum(1 for x in x_test if len(x)>=length))
        print("dev", sum(1 for x in x_dev if len(x)>=length))

def test_dictionary():
    import json

    version = "LSTM"
    ml_model_dir = os.path.join(model_dir, version)

    with open(os.path.join(ml_model_dir, "dictionary.model"), 'rb') as infile:
        model = joblib.load(infile)

    with open(os.path.join(ml_model_dir, "dictionary.json"), 'w', encoding='utf-8') as outfile:
        json.dump(model.dictionary, outfile, indent=4)

def parse_arg():
    parser = argparse.ArgumentParser(description="Classification Baseline.")
    parser.add_argument("--model", dest="model", help="SVM/RandomForest/CNN/LSTM", type=str, default="SVM")
    return parser.parse_args()

def main():
    arg = parse_arg()
    if arg.model in {"SVM", "RandomForest"}:
        ml_baseline(arg)
    if arg.model in {"LSTM", "CNN"}:
        dl_baseline(arg)

def test():
    check_length()
    test_dictionary()

if __name__ == "__main__":
    main()

