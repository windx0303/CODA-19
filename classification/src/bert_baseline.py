from data import load_data
from config import *
from util import *
from baseline import CovidDataset, evaluate
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import copy

from transformers import BertTokenizerFast, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, PretrainedConfig, BertConfig, AutoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils import data
import numpy as np
import os

from pprint import pprint
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Feature:
    def __init__(self, pad_length=100, tokenizer=None):
        self.pad_length = pad_length
        if tokenizer is None:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer
        self.pad_id, self.cls_id, self.sep_id = self.tokenizer.convert_tokens_to_ids(
                [self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.sep_token]
            )

    def extract(self, sents):
        results = [self.tokenizer.encode(s, add_special_tokens=False) for s in sents]

        # turn to matrix with padding
        matrix = np.ones([len(results), self.pad_length], dtype=np.int32) * self.pad_id
        for i, res in enumerate(results):
            length = min(len(res), self.pad_length)
            matrix[i, :length] = res[:length]

        cls_matrix = np.ones([len(results), 1]) * self.cls_id
        sep_matrix = np.ones([len(results), 1]) * self.sep_id
        matrix = np.hstack([cls_matrix, matrix, sep_matrix])
        
        return matrix

class CovidDataset(data.Dataset):
    def __init__(self, x, y, bucket_num=None):
        self.x = x.astype(np.int64)
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.y.shape[0] 

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def evaluate(model, eval_data, device):
    model.eval()
    total_count = len(eval_data.dataset) // eval_data.batch_size
    total_loss = 0
    total_acc = 0
    predict = []
    true_label = []
    with torch.no_grad():
        for count, (x_batch, y_batch) in enumerate(eval_data, 1):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch, labels=y_batch)
            loss, y_pred = outputs[0:2]

            # compute loss
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

def bert_baseline(arg):
    from bert_config import bert_parameter_dict
    version = arg.model

    parameters          = bert_parameter_dict[version]
    batch_size          = parameters["batch_size"]
    epoch_num           = parameters["epoch_num"]
    learning_rate       = parameters["learning_rate"]
    device              = parameters["device"]
    early_stop_epoch    = parameters["early_stop_epoch"]

    dl_model_dir = os.path.join(model_dir, version)
    create_dir(dl_model_dir)

    data_cached_path = os.path.join(cache_dir, version+".h5")
    if os.path.isfile(data_cached_path):
        x_train, y_train, x_test, y_test, x_dev, y_dev = h5_load(data_cached_path, [
            "x_train", "y_train", "x_test", "y_test", "x_dev", "y_dev"
        ], dtype=np.int32, verbose=True)

    else:
        # load data
        x_train, y_train = load_data(phrase="train", verbose=True)
        x_test, y_test = load_data(phrase="test", verbose=True)
        x_dev, y_dev = load_data(phrase="dev", verbose=True)
        
        # turn text into ids
        if version == "bert":
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        elif version == "sci-bert":
            tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        tokenizer.save_pretrained(dl_model_dir)

        feature = Feature(tokenizer=tokenizer)
        x_train = feature.extract(x_train[:])
        x_test = feature.extract(x_test[:])
        x_dev = feature.extract(x_dev[:])

        # turn label into vector
        y_train = np.array([label_mapping[y] for y in y_train])
        y_test = np.array([label_mapping[y] for y in y_test])
        y_dev = np.array([label_mapping[y] for y in y_dev])

        # cache data
        with h5py.File(data_cached_path, 'w') as outfile:
            outfile.create_dataset("x_train", data=x_train) 
            outfile.create_dataset("y_train", data=y_train)
            outfile.create_dataset("x_test", data=x_test) 
            outfile.create_dataset("y_test", data=y_test)
            outfile.create_dataset("x_dev", data=x_dev) 
            outfile.create_dataset("y_dev", data=y_dev)

    print("Train", x_train.shape, y_train.shape)
    print("Test", x_test.shape, y_test.shape)
    print("Valid", x_dev.shape, y_dev.shape)

    #subset_num = 1000
    #x_train, y_train = x_train[:subset_num], y_train[:subset_num]
    #x_dev, y_dev = x_dev[:subset_num], y_dev[:subset_num]
    #x_test, y_test = x_test[:subset_num], y_test[:subset_num]

    train_dataset = CovidDataset(x_train, y_train)
    test_dataset = CovidDataset(x_test, y_test)
    dev_dataset = CovidDataset(x_dev, y_dev)
    training = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    testing = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    dev = data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # model
    if version == "bert":
        print("Using Bert!!!")
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5).to(device)
    elif version == "sci-bert":
        print("Using SCI-Bert!!!")
        #config = BertConfig(vocab_size=31090, num_labels=5)
        config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased')
        config.num_labels = 5
        model = AutoModelForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', config=config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    acc, _, _ = evaluate(model, dev, device=device)
    best_model = None
    best_accuracy = 0.0
    best_epoch = 0
    stopper = EarlyStop(mode="max", history=early_stop_epoch)

    for epoch in range(1, epoch_num+1):
        model.train()
        total_loss = 0
        total_acc = 0
        total_count = len(train_dataset) // batch_size
        for count, (x_batch, y_batch) in enumerate(training, 1):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch, labels=y_batch)
            loss, y_pred = outputs[0:2]
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

            # check early stopping
            if stopper.check(acc):
                print("Early Stopping at Epoch = ", epoch)
                break

    # load best model & test & save
    print("loading model from epoch {}".format(best_epoch))
    #torch.save(best_model, os.path.join(dl_model_dir, "best_model.pt"))
    model.load_state_dict(best_model)
    model.save_pretrained(dl_model_dir)
    acc, predict, true_label = evaluate(model, testing, device=device) 
    score = precision_recall_fscore_support(true_label, predict)
    table = output_score(score)
    print(table)

    # output result
    with open(os.path.join(result_dir, "{}.result".format(version)), 'w', encoding='utf-8') as outfile:
        outfile.write(table.to_csv(path_or_buf=None)+"\n")
        outfile.write("acc = {}\n".format(acc))

def check_length():
    # load data
    x_train, y_train = load_data(phrase="train", verbose=True)
    x_test, y_test = load_data(phrase="test", verbose=True)
    x_dev, y_dev = load_data(phrase="dev", verbose=True)
    
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    x_train = [tokenizer.encode(s, add_special_tokens=False) for s in x_train]
    x_test = [tokenizer.encode(s, add_special_tokens=False) for s in x_test]
    x_dev = [tokenizer.encode(s, add_special_tokens=False) for s in x_dev]

    # check length
    for length in [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]:
        print()
        print(length)
        print("train", sum(1 for x in x_train if len(x)>=length))
        print("test", sum(1 for x in x_test if len(x)>=length))
        print("dev", sum(1 for x in x_dev if len(x)>=length))

def parse_arg():
    parser = argparse.ArgumentParser(description="Classification BERT Baseline.")
    parser.add_argument("--model", dest="model", help="bert/sci-bert", type=str, default="bert")
    return parser.parse_args()

def main():
    arg = parse_arg()
    bert_baseline(arg)

if __name__ == "__main__":
    main()
