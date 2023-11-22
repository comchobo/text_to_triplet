import os
import pandas as pd
from tqdm import tqdm
from Filter_N_Translate.Preprocessor import Preprocessor
from transformers import AutoTokenizer
preprocessor = Preprocessor()

# set_korean_news
path = './english_news'
tokenizer = AutoTokenizer.from_pretrained('microsoft/mdeberta-v3-base')

import json
with open('./filtered_idxs.json','r')as f:
    idxs_dict = json.load(f)

def put_till_n(text_list, n):
    temp = text_list[0]
    if len(text_list)==1 : return temp
    for text in text_list[1:]:
        if len(temp)<n:
            temp+=text
        else:
            return temp
    return temp

years = [x for x in range(2012, 2023)]
news_text = []
news_title = []
labels=[]
for year in years:
    svos = []
    files_of_years = [x for x in os.listdir(path) if x[:4] == str(year)]
    for files in tqdm(files_of_years):
        csv_file = pd.read_csv(f'{path}/{files}', encoding='UTF8')

        for i in range(len(csv_file)):
            news_text_temp = csv_file.iloc[i].text
            preprocessed_news_text_temp = preprocessor.split_raw_text_into_paragraph(news_text_temp)
            if isinstance(preprocessed_news_text_temp, float) == True \
                    or len(preprocessed_news_text_temp)<1 :
                continue

            if files in idxs_dict:
                if i in idxs_dict[files]:
                    news_text.append(put_till_n(preprocessed_news_text_temp, 500))
                    news_title.append(csv_file.iloc[i].title)
                    labels.append(1)
                else:
                    news_text.append(put_till_n(preprocessed_news_text_temp, 500))
                    news_title.append(csv_file.iloc[i].title)
                    labels.append(0)
            else:
                news_text.append(put_till_n(preprocessed_news_text_temp, 500))
                news_title.append(csv_file.iloc[i].title)
                labels.append(0)

from datasets import Dataset
def preprocess_function(examples):
    model_inputs = tokenizer(examples['title'], examples['text'], max_length=128, truncation=True)
    return model_inputs

dataset = Dataset.from_dict({'title':news_title, 'text':news_text, 'labels':labels})
tokenized_datasets = dataset.map(preprocess_function, batched=True, batch_size=1000)
tokenized_datasets.save_to_disk('./L_tokenized_english_news')