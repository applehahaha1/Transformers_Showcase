# config.py
import transformers

# this is the maximum number of tokens in a single sentence
MAX_LEN = 512

# batch size is small because model is huge
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4

# train for 10 epochs
EPOCHS = 10

# define path to BERT model files
BERT_PATH = "./input/bert-base-uncased/"
# BERT_PATH = "bert-base-uncased"

# model name
# MODEL_PATH = "model.bin"
MODEL_PATH = "model..safetensors"

# define path to dataset
TRAINING_FILE = "./input/imdb.csv"

# define tokenizer
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)