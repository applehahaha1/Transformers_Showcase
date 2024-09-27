<<<<<<< HEAD
# Transformers Showcase

This repository contains a showcase of transformer-based models for natural language processing tasks.

## Dataset

The dataset used in this repository is the IMDB dataset, which is a collection of movie reviews with corresponding sentiment labels (positive or negative).

## Configuration

The configuration for the models is stored in the src/config.py 
file. This file defines the following constants:

* `MAX_LEN`: The maximum length of a single sentence.
* `TRAIN_BATCH_SIZE`: The batch size for training.
* `VALID_BATCH_SIZE`: The batch size for validation.
* `EPOCHS`: The number of epochs to train for.
* `BERT_PATH`: The path to the pre-trained BERT model.
* `MODEL_PATH`: The path to save the trained model.
* `TRAINING_FILE`: The path to the training data.
* `TOKENIZER`: The tokenizer to use for preprocessing the text data.

## Model

The model used in this repository is a BERT-based model for sentiment analysis. The model is defined in the [src/model.py] file.

## Training

The training script is stored in the `train.py` file. This script trains the model using the IMDB dataset and saves the trained model to the `model.bin` file.

## Evaluation

The evaluation script is stored in the `eval.py` file. This script evaluates the trained model using the IMDB dataset and prints the accuracy and F1 score.

## Usage

To train the model, run the following command:
```bash
python train.py
=======
# Transformers Showcase

This repository contains a showcase of transformer-based models for natural language processing tasks.

## Dataset

The dataset used in this repository is the IMDB dataset, which is a collection of movie reviews with corresponding sentiment labels (positive or negative).

## Configuration

The configuration for the models is stored in the src/config.py 
file. This file defines the following constants:

* `MAX_LEN`: The maximum length of a single sentence.
* `TRAIN_BATCH_SIZE`: The batch size for training.
* `VALID_BATCH_SIZE`: The batch size for validation.
* `EPOCHS`: The number of epochs to train for.
* `BERT_PATH`: The path to the pre-trained BERT model.
* `MODEL_PATH`: The path to save the trained model.
* `TRAINING_FILE`: The path to the training data.
* `TOKENIZER`: The tokenizer to use for preprocessing the text data.

## Model

The model used in this repository is a BERT-based model for sentiment analysis. The model is defined in the [src/model.py] file.

## Training

The training script is stored in the `train.py` file. This script trains the model using the IMDB dataset and saves the trained model to the `model.bin` file.

## Evaluation

The evaluation script is stored in the `eval.py` file. This script evaluates the trained model using the IMDB dataset and prints the accuracy and F1 score.

## Usage

To train the model, run the following command:
```bash
python train.py
>>>>>>> 0e15cc6646b485e850abac929b70f26ba82cc2d0
