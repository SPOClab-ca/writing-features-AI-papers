import torch
import transformers


class ListDataset(torch.utils.data.Dataset):
  """
  Python list of form [(sent1, label1), (sent2, label2), ...]
  """

  def __init__(self, data):
    self.data = data
    self.tokenizer = transformers.RobertaTokenizerFast.from_pretrained('roberta-base')


  def __len__(self):
    return len(self.data)
  

  def __getitem__(self, ix):
    cur_str, cur_label = self.data[ix]
    tok_result = self.tokenizer(cur_str, padding='max_length', max_length=200, truncation=True)
    return torch.tensor(tok_result['input_ids']), torch.tensor(tok_result['attention_mask']), cur_label
