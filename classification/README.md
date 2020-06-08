# Classification Baseline
This is the implementation of the baseline models for the classification task using CODA-19 dataset.

To run the experiment, please change the **root_dir** in **src/config.py** to the **CODA-19 folder** in your machine.
```python
root_dir = "/home/appleternity/workspace/lab/Crowd/CODA-19"
```



## Compute Agreement
```console
$ cd src
$ python compute_agreement.py
```

## Run baseline model
Four baseline models are implemented, SVM, RandomForest, LSTM, and CNN.
The default model is SVM, if you would like to run other models, please specify it using **--model**.
Experiment results and trained models will be stored in the **result** and **model** folder respectively. 
```
$ cd src
$ python baseline.py [--model MODEL_NAME]
```

## Run BERT baseline model
Two bert models are implemented, BERT and Sci-BERT, using [HuggingFace's implementation](https://github.com/huggingface/transformers).
The default model is BERT, if you would like to run other models, please specify it using **--model**.
Experiment results and trained models will be stored in the **result** and **model** folder respectively.
Since BERT has its own tokenizer, the tokenized and processed data will be stored in the **cache** folder.
```
$ cd src
$ python bert_baseline [--model MODEL_NAME]
```

#### If you would like to change the hyper-parameters...

Please modify **ml_config.py** for SVM and RandomForest. 

Modify **dl_config.py** for LSTM and CNN.

Modify **bert_config.py** for BERT and Sci-BERT.

