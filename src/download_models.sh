# !/bin/bash
# This script offers a limited selection of model examples;
# please feel free to customize it according to your requirements.
# reference: https://huggingface.co/models 

# Assume you are located within the 'models/' directory. 

# =======
# BERT
# =======
mkdir bert-base-uncased
cd bert-base-uncased
wget https://huggingface.co/bert-base-uncased/resolve/main/config.json
wget https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin
wget https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt

cd ..

mkdir bert-base-cased
cd bert-base-cased
wget https://huggingface.co/bert-base-cased/resolve/main/config.json
wget https://huggingface.co/bert-base-cased/resolve/main/pytorch_model.bin
wget https://huggingface.co/bert-base-cased/resolve/main/vocab.txt

cd ..

# =======
# RoBERTa
# =======
mkdir roberta-base
cd roberta-base
wget https://huggingface.co/roberta-base/resolve/main/config.json
wget https://huggingface.co/roberta-base/resolve/main/pytorch_model.bin
wget https://huggingface.co/roberta-base/resolve/main/vocab.json
wget https://huggingface.co/roberta-base/resolve/main/merges.txt

cd ..

# =======
# ALBERT
# =======
mkdir albert-base-v2
cd albert-base-v2
wget https://huggingface.co/albert-base-v2/resolve/main/config.json
wget https://huggingface.co/albert-base-v2/resolve/main/pytorch_model.bin
wget https://huggingface.co/albert-base-v2/resolve/main/spiece.model

cd .. 

# =======
# DistilBERT
# =======
mkdir distilbert-base-uncased
cd distilbert-base-uncased
wget https://huggingface.co/distilbert-base-uncased/resolve/main/config.json
wget https://huggingface.co/distilbert-base-uncased/resolve/main/pytorch_model.bin
wget https://huggingface.co/distilbert-base-uncased/resolve/main/vocab.txt

# Usage example:
# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('./bert-base-cased')
# model = BertModel.from_pretrained('./bert-base-cased')