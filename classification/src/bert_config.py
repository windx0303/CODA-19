BERT_parameter = {
    "batch_size": 32,
    "epoch_num": 100,
    "learning_rate": 1e-7,
    "device": "cuda:0",
    "early_stop_epoch": 5,
}

SCIBERT_parameter = {
    "batch_size": 32,
    "epoch_num": 100,
    "learning_rate": 1e-7,
    "device": "cuda:0",
    "early_stop_epoch": 5,
}

bert_parameter_dict = {
    "bert": BERT_parameter,
    "sci-bert": SCIBERT_parameter,
}
