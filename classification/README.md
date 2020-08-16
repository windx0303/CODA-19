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
```console
$ cd src
$ python baseline.py [--model MODEL_NAME]
```

## Run BERT baseline model
Two bert models are implemented, BERT and Sci-BERT, using [HuggingFace's implementation](https://github.com/huggingface/transformers).
The default model is BERT, if you would like to run other models, please specify it using **--model**.
Experiment results and trained models will be stored in the **result** and **model** folder respectively.
Since BERT has its own tokenizer, the tokenized and processed data will be stored in the **cache** folder.
```console
$ cd src
$ python bert_baseline.py [--model MODEL_NAME]
```

#### If you would like to change the hyper-parameters...

Please modify **ml_config.py** for SVM and RandomForest. 

Modify **dl_config.py** for LSTM and CNN.

Modify **bert_config.py** for BERT and Sci-BERT.

## Use the fine-tuned BERT and Sci-BERT model

**Now you can access the model directly from HuggingFace!**

#### BERT
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("appleternity/bert-base-uncased-finetuned-coda19")
model = AutoModelForSequenceClassification.from_pretrained("appleternity/bert-base-uncased-finetuned-coda19")
```

#### Sci-BERT
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("appleternity/scibert-uncased-finetuned-coda19")
model = AutoModelForSequenceClassification.from_pretrained("appleternity/scibert-uncased-finetuned-coda19")
```



**! Below are the old method**

The fine-tuned BERT and Sci-BERT model can be found [**here**](https://drive.google.com/drive/folders/1o75jKKBScu2AOoEUxHIuxbEz0BmZxs-2?usp=sharing).

URL: https://drive.google.com/drive/folders/1o75jKKBScu2AOoEUxHIuxbEz0BmZxs-2?usp=sharing

You can use the following code to load the model.

#### BERT
```python
from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
model.load_state_dict(torch.load("bert_best_model.pt"))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
```
#### Sci-BERT
```python
from transformers AutoTokenizer, AutoModelForSequenceClassification, BertConfig

config = BertConfig(vocab_size=31090, num_labels=5)
model = AutoModelForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', config=config)
model.load_state_dict(torch.load("scibert_best_model.pt"))
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
```

