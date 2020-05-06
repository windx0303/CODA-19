LSTM_parameter = {
    "batch_size": 256,
    "hidden_size": 256,
    "layer_num": 10,
    "epoch_num": 50,
    "learning_rate": 0.00005,
    "reg_weight": 1e-6,
    "dropout_rate": 0.3,
    "device": "cuda",
}

CNN_parameter = {
    "batch_size": 256,
    "hidden_size": 256,
    "filter_list": [(3, 100), (4, 100), (5, 100)],
    "epoch_num": 50,
    "learning_rate": 0.00005,
    "reg_weight": 1e-6,
    "dropout_rate": 0.3,
    "device": "cuda",
}

dl_parameter_dict = {
    "LSTM": LSTM_parameter,
    "CNN": CNN_parameter,
}
