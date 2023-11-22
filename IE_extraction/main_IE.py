import datasets.config
from EntityExtracter import EntityExtracterModule
from RelationMatcher import RelationMatcherModule
from InferenceNLI import InferenceNLIModule

import os
import pandas as pd
import torch
from datasets import Dataset, disable_progress_bar
from tqdm import tqdm, trange

if __name__ == '__main__' :
    device = torch.device("cuda:0")
    preprocessed_path = '../result_1/1_2_3_4__S'
    target_years = [str(x) for x in range(2020,2023)]
    files_of_years = [f'{preprocessed_path}/{x}' for x in os.listdir(preprocessed_path) if x[:4] in target_years]

    import random
    files_of_years = ['blinded']
    files_of_years = [f'{preprocessed_path}/{x}' for x in files_of_years]

    concated_news = pd.concat(map(pd.read_csv, tqdm(files_of_years)))
    concated_dataset = Dataset.from_pandas(concated_news)
    concated_dataset = concated_dataset.remove_columns('__index_level_0__')
    del concated_news

    # EntityExtracter로 Entity 생성
    onto_path = 'blinded.owl'
    entity_extractor = EntityExtracterModule(onto_path, pos_model_path="TweebankNLP/bertweet-tb2_ewt-pos-tagging"
                 , ner_model_path='Gladiator/microsoft-deberta-v3-large_ner_conll2003')
    entity_extractor.load_model_on_device(device)
    entities = entity_extractor.extract_entities(concated_dataset)

    # RelationMatcher로 acceptable한 relation들 생성
    relation_matcher = RelationMatcherModule(onto_path, threshold=0.78)
    relation_matcher.load_model_on_device(device)
    relation_matcher.set_output_path('../result_2/IE_v2')
    res = relation_matcher.extract_relations_batch(concated_dataset, entities)
