from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

path='../deep_models/lora_training_sum_t5_lr=8e-4_saved'

import os
alter_path = 'd:/dataset/new_korean_news'
target_years = [str(x) for x in range(2020, 2023)]
files_of_years = [f'{alter_path}/{x}' for x in os.listdir(alter_path) if x[:4] in target_years]

import random
random.seed(42)
files_of_years = random.sample(files_of_years, 1) # for debug

# 'summarize: '
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
concated_news = pd.concat(map(pd.read_csv, tqdm(files_of_years)))
concated_dataset = Dataset.from_pandas(concated_news)
concated_dataset = concated_dataset.rename_column('Unnamed: 0', 'original_index')

config = PeftConfig.from_pretrained(path)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, src_lang="eng_Latn")
inputs = tokenizer(article, return_tensors="pt", padding=True)

translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["kor_Hang"], max_length=200
)
for sent in tokenizer.batch_decode(translated_tokens, skip_special_tokens=True):
    print(sent)
