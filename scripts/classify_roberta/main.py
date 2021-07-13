import argparse
import torch
import transformers
import pickle
import random
import sklearn.model_selection

import dataloader



def load_data(venue):
  global train_data, test_data

  with open('../../data/text_classify_articles_with_arxiv.pkl', 'rb') as f:
    data = pickle.load(f)

  naacl_data = data[venue]
  naacl_data = [(paper['abstract'], paper['label']) for paper in naacl_data]
  labels = [item[1] for item in naacl_data]

  # 80/20 train-test split
  train_data, test_data = sklearn.model_selection.train_test_split(
    naacl_data, test_size=0.2, random_state=args.seed, stratify=labels
  )
  #print ("Loaded venue", venue)


def train():
  global model
  #print("Training...")
  train_loader = torch.utils.data.DataLoader(
    dataloader.ListDataset(train_data), batch_size=16, shuffle=True
  )

  model = transformers.RobertaForSequenceClassification.from_pretrained('roberta-base').cuda()
  model.train()
  opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

  for epoch in range(10):
    step = 0
    for tok_ids, attn_masks, labels in train_loader:
      tok_ids = tok_ids.cuda()
      attn_masks = attn_masks.cuda()
      labels = labels.cuda()

      opt.zero_grad()
      outputs = model(tok_ids, attn_masks, labels=labels)
      loss = outputs.loss
      loss.backward()
      opt.step()

      step += 1
      #print(f"Epoch {epoch}, step {step}, loss={float(loss)}")

    print ("Epoch", epoch, end="\t")
    evaluate()


def evaluate():
  #print("Evaluating...")
  model.eval()
  true_labels = [instance[1] for instance in test_data]
  eval_loader = torch.utils.data.DataLoader(
    dataloader.ListDataset(test_data), batch_size=16
  )

  predictions = []
  for tok_ids, attn_masks, labels in eval_loader:
    tok_ids = tok_ids.cuda()
    attn_masks = attn_masks.cuda()

    outputs = model(tok_ids, attn_masks)
    pred = torch.max(outputs.logits, 1).indices
    predictions.extend(pred.cpu().tolist())

  acc_score = sklearn.metrics.accuracy_score(true_labels, predictions)
  f1_score = sklearn.metrics.f1_score(true_labels, predictions)
  #print('True labels:', true_labels)
  #print('Predictions:', predictions)
  print('Accuracy:', acc_score, end=", ")
  print('F1 score:', f1_score)
  model.train()


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--venue", type=str)
args = parser.parse_args() 
print(args)


random.seed(args.seed)
torch.manual_seed(args.seed)
load_data(args.venue)
train()
evaluate()
