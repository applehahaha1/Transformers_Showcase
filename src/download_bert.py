from transformers import BertModel, BertTokenizer

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Save the tokenizer and model
tokenizer.save_pretrained("./input/bert-base-uncased")
model.save_pretrained("./input/bert-base-uncased")
