# Text Retrieval on Cranfield dataset with Vector Space Model.

We apply some preprocessing for text and then use n_ngram for token text. To generate vector, we use two way, first is TF-IDF and the second is LSI. Finally, we compute the similarity of text by cosine distance and return MAP score for corpus TEST set.

## 1. Setup

### 1.1 Install packages:
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 1.2 Data folder:
- Data train: Cranfield folder

- Data test: TEST/Query.txt

- Ground Truth: TEST/RES


## 2. Usage:

### 2.1 Predict
```
python main.py --path_docs --path_query --path_folder_save --topk --model
```
Example:
Use TF-IDF to create vector:
```
python main.py --topk 1400 --model vsm
```
path_docs, path_query, path_folder_save should be default

### 2.2 Evaluate
```
python compute_map.py
```

