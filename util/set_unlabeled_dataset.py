import os
import pandas as pd
from tqdm import tqdm
from Filter_N_Translate.Preprocessor import Preprocessor
from transformers import AutoTokenizer
from datasets import Dataset
preprocessor = Preprocessor()
tokenizer = AutoTokenizer.from_pretrained('microsoft/mdeberta-v3-base')

def preprocess_function(examples):
    model_inputs = tokenizer(examples['title'], examples['text'], max_length=128, truncation=True, padding=True)
    return model_inputs

# set_korean_news
path = './korean_news'

years = [x for x in range(2012, 2024)]
news_data = []
title_data = []
for year in years:
    svos = []
    files_of_years = [x for x in os.listdir(path) if x[:4] == str(year)]
    for files in tqdm(files_of_years):
        csv_file = pd.read_csv(f'{path}/{files}', encoding='UTF8')
        for i in range(len(csv_file)):
            if isinstance(csv_file.iloc[i].title, float) == True or len(csv_file.iloc[i].title)<2:
                continue
            news_text_temp = csv_file.iloc[i].text
            if preprocessor.preprocess_korean_news(news_text_temp) == 'error' or len(news_text_temp) < 20:
                continue
            title_data.append(csv_file.iloc[i].title)
            news_data.append(news_text_temp[:750])

dataset = Dataset.from_dict({'title':title_data,'text':news_data})
tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets.save_to_disk('./UL_tokenized_korean_news')

# set_twitter_data

path = './tweets'

# import re
# years = [x for x in range(2012, 2024)]
# tweets_data = []
# for year in years:
#     svos = []
#     files_of_years = [x for x in os.listdir(path) if x[:4] == str(year)]
#     for files in tqdm(files_of_years):
#         csv_file = pd.read_csv(f'{path}/{files}', encoding='UTF8')
#         for i in range(len(csv_file)):
#             tweets_temp = csv_file.iloc[i].text
#             tweets_temp= re.sub('https://t\.co.*','',tweets_temp)
#             tweets_temp = re.sub('http://t\.co.*', '', tweets_temp)
#             if len(tweets_temp) < 10:
#                 continue
#             tweets_data.append(tweets_temp)
#
# dataset = Dataset.from_dict({'text':tweets_data})
# tokenized_datasets = dataset.map(preprocess_function, batched=True, batch_size=10000)
# tokenized_datasets.save_to_disk('./UL_tokenized_tweets')

# set_english_news -> suspended
