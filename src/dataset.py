#dataset.py
import config
import torch 

class BERTDataset:
    def __init__(self,review,target):
        """
        Initializes a BERTDataset instance.

        Parameters:
            self (BERTDataset): The instance being initialized.
            review (str): The review text.
            target (str): The target label for the review.

        Returns:
            None
        """
        self.review = review    
        self.target = target

        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        """
        Returns the length of the dataset.

        Parameters:
            self (BERTDataset): The instance being used.

        Returns:
            int: The length of the dataset.
        """
        return len(self.review)

    def __getitem__(self, item):
        """
        Returns the item at the specified index.

        Parameters:
            self (BERTDataset): The instance being used.
            item (int): The index of the item to return.

        Returns:
            dict: A dictionary containing the review text and the target label.
        """
        review = str(self.review[item])
        target = self.target[item]

        # encode_plus comes from huggingface's transformers library,
        # and exists for all tokenizers they offer.
        # It can be used to convert a given string to ids, masks and token type ids
        # which are needed for models like BERT here, review is a string
        inputs = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )

        # ids are ids of tokens generated after tokenizing reviews
        ids = inputs['input_ids']

        # mask is 1 where we have input and 0 where we have padding
        mask = inputs['attention_mask']

        # token type ids behave the same way as mask in this specific case
        # in case of two sentences, this is 0 for first sentence and 1 for second sentence
        token_type_ids = inputs['token_type_ids']

        # note that ids, mask and token_type_ids are all long datatypes and targets is float
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dytype= torch.long),
            'target': torch.tensor(target, dtype=torch.long)
        }