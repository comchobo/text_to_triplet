from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch, json, re, ast
import numpy as np
device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained('microsoft/mdeberta-v3-base')
model = AutoModelForSequenceClassification.from_pretrained('./classifier')
import datasets
dataset = datasets.load_from_disk('UL_tokenized_korean_news')

res = []
model.to(device)

chunker = []
temp = []
for x in range(len(dataset)):
    if x%256==0:
        chunker.append(temp)
        temp = []
    else:
        temp.append(x)

acc = []
labeled_text = []
from transformers import BatchEncoding
from tqdm import tqdm
for chunk in tqdm(chunker[1:100]):
    data = dataset[chunk]
    temp = {'title': data['title'], 'text': data['text']}
    data = BatchEncoding({'input_ids':torch.tensor(data['input_ids']),'token_type_ids':torch.tensor(data['token_type_ids'])
                             ,'attention_mask':torch.tensor(data['attention_mask'])})
    with torch.no_grad():
        data.to(device)
        predicted = model(**data)
    predicted_title_label = np.argmax(predicted.logits.detach().cpu().numpy(), axis=-1)
    for idx in np.where(predicted_title_label==1)[0]:
        labeled_text.append([temp['title'][idx], temp['text'][idx]])
    acc += predicted_title_label.tolist()

print(labeled_text)