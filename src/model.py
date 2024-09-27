#model.py
import config
import transformers
import torch.nn as nn

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        # load pretrained BERT model
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        # dropout layer
        self.bert_drop = nn.Dropout(0.3)
        # dense layer
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        # get output from BERT model
        # BERT in its default settings returns two outputs
        # last hidden state and output of bert pooler layer
        # we use the output of the pooler which is of the size 
        # (batch_size, hidden_size)
        # hidden size can be 768 or 1024 depending on
        # if we are using bert base or large respectively
        # in our case, it is 768
        # note that this model is pretty simple
        # you might want to use last hidden state
        # or several hidden states
        res = self.bert(ids, 
                        attention_mask=mask, 
                        token_type_ids=token_type_ids)        # dropout
        o2 = res['pooler_output']
        bo = self.bert_drop(o2)
        # output
        output = self.out(bo)
        return output