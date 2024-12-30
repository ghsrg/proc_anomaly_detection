# Transformers : BERT-like Model (using Hugging Face Transformers)
from transformers import BertModel
from torch import nn

class TransformerClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_dim=768, num_labels=2):
        super(TransformerClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.fc = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token output
        logits = self.fc(cls_output)
        return logits