from Filter_N_Translate.Preprocessor import DeepPreprocessorModule
from Filter_N_Translate.NewsFilteringModule import NewsFilteringModule
from Filter_N_Translate.SentenceSplitter import SentenceSplitterModule
from Filter_N_Translate.LanguageTranslator import LanguageTranslateModule
from Filter_N_Translate.DuplicateNewsDetector import DuplicateNewsDetectorModule
from Filter_N_Translate.EntityLinkedLanguageTranslator import EntityLinkedLanguageTranslateModule

import os, json, gc
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm


def preprocess(dataset, args):
    preprocessor = DeepPreprocessorModule(num_workers=args.num_worker, model_path=args.preprocessing_model_path
                                          , tokenizer_path = args.preprocessing_tokenizer_path, lang_mode=args.lang_mode
                                          , save_results=True, check_time=True)
    preprocessor.set_output_path(f'{args.output_path}/1')
    preprocessor.set_device(args.device)
    preprocessed_concated_news = preprocessor.deep_preprocess(dataset)
    return preprocessed_concated_news


def filter(dataset, args):
    newsfilterer = NewsFilteringModule(num_workers=args.num_worker, model_path=args.filtering_model_path
                                       , tokenizer_path = args.filtering_tokenizer_path, lang_mode=args.lang_mode
                                       , save_results=True, check_time=True)
    newsfilterer.set_output_path(f'{args.output_path}/1_')
    newsfilterer.set_device(args.device)
    filtered_concated_news = newsfilterer.deep_filter(dataset, filtering_thres=-0.4)
    return filtered_concated_news


def erase_similar(dataset, args):
    duperaser = DuplicateNewsDetectorModule(num_workers=args.num_worker, model_path=args.eraser_model_path
                                            , tokenizer_path=args.eraser_model_path, lang_mode=args.lang_mode
                                            , save_results=True, check_time=True)
    duperaser.set_output_path(f'{args.output_path}/1_2_')
    duperaser.set_device(args.device)
    dup_erased_dataset = duperaser.erase_similar_and_duplicated(dataset, filtering_thres=0.35)
    return dup_erased_dataset


def split(dataset, args):
    splitter = SentenceSplitterModule(num_workers=args.num_worker, model_path=None
                                      , tokenizer_path = None, lang_mode=args.lang_mode)
    splitter.set_output_path(f'{args.output_path}/1_2_3(SPL)')
    sentence_split_dataset = splitter.split_sentence(dataset)
    return sentence_split_dataset


def translate(dataset, args):
    translator = LanguageTranslateModule(num_workers=args.num_worker, model_path=args.translation_model_path,
                                          tokenizer_path = args.translation_model_path)
    translator.set_output_path(f'{args.output_path}/1_2_3(SPL)_4')
    translator.set_device(args.device)
    translated_dataset = translator.deep_translate(dataset, SPL=1)
    return translated_dataset


def entity_linked_translate(dataset, args):
    torch.cuda.empty_cache()
    translator = EntityLinkedLanguageTranslateModule(num_workers=args.num_worker, model_path=args.translation_model_path,
                                          tokenizer_path = args.translation_model_path)
    translator.set_output_path(f'{args.output_path}/1_2_3(SPL)_4(EL)')
    translator.set_device(args.device)
    translated_dataset = translator.deep_translate(dataset, SPL=1)
    return translated_dataset


def main(args, task_list):
    import sys
    sys.path.append("/")
    device = torch.device("cuda")

    year_range = args.year_range.split('-')
    if year_range[0] == year_range[1]:
        target_years = [year_range[0]]
    else:
        target_years = [str(x) for x in range(int(year_range[0]), int(year_range[1]))]

    for target_year in target_years:
        files_of_years = [f'{args.input_path}/{x}' for x in os.listdir(args.input_path) if x[:4] == target_year]

        if args.debug_mode:
            import random
            random.seed(111)
            files_of_years = random.sample(files_of_years, 3) # for debug
            num_worker = 1
        else:
            num_worker = 8

        if 'csv' in files_of_years[0]:
            concated_news = pd.concat(map(pd.read_csv, tqdm(files_of_years)))
            try:
                concated_news = concated_news.drop('__index_level_0__', axis=1)
            except KeyError:
                pass

            concated_dataset = Dataset.from_pandas(concated_news)
            try:
                concated_dataset = concated_dataset.rename_column('Unnamed: 0', 'original_index')
            except ValueError:
                pass
            del concated_news
        else:
            from datasets import load_dataset
            concated_dataset = load_dataset("json", data_files=files_of_years)['train']

        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_path', default=args.input_path)
        parser.add_argument('--output_path', default=args.output_path)
        parser.add_argument('--num_worker', default=num_worker)
        parser.add_argument('--device', default=device)

        if args.lang_mode == 'KOR':
            parser.add_argument("--preprocessing_model_path", default="embargo") # Company owns the model
            parser.add_argument("--preprocessing_tokenizer_path", default="embargo") # Company owns the model
            parser.add_argument("--filtering_model_path", default="embargo") # Company owns the model
            parser.add_argument("--filtering_tokenizer_path", default="microsoft/mdeberta-v3-base")
            parser.add_argument("--eraser_model_path", default="embargo") # Company owns the model
            parser.add_argument("--translation_model_path", default="embargo") # Company owns the model
            parser.add_argument("--lang_mode", default=args.lang_mode)
        elif args.lang_mode == 'ENG':
            # parser.add_argument("--preprocessing_model_path", default='deep_models/saved_fixed_preprocessing_model2')  # local!
            # parser.add_argument("--preprocessing_tokenizer_path", default='deep_models/saved_fixed_preprocessing_model2')  # local!
            parser.add_argument("--preprocessing_model_path",
                                default='deep_models/saved_fixed_preprocessing_model2')  # local model
            parser.add_argument("--preprocessing_tokenizer_path",
                                default='deep_models/deep_preprocess_tokenizer')  # local model

            parser.add_argument("--filtering_model_path", default="embargo") # Company owns the model
            parser.add_argument("--filtering_tokenizer_path", default="microsoft/mdeberta-v3-base")
            parser.add_argument("--eraser_model_path", default='sentence-transformers/all-mpnet-base-v2')
            parser.add_argument("--lang_mode", default=args.lang_mode)
        else:
            print('Error : lang_mode not selected!')
            exit()

        args_for_modules = parser.parse_args()

        res = concated_dataset
        for task in task_list:
            res = globals()[task](res, args_for_modules)
        del res
        del concated_dataset
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__' :
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',default='result_1_v6(KR)/1_2_3(SPL)')
    parser.add_argument('--output_path',default='result_1_v6(KR)')

    parser.add_argument('--year_range',default='2017-2024')
    parser.add_argument('--debug_mode', default=False)
    parser.add_argument('--lang_mode', default='KOR')

    # task_list = ['preprocess', 'filter', 'erase_similar', 'split', 'translate', 'entity_linked_translate']
    task_list = ['entity_linked_translate']

    args = parser.parse_args()
    main(args, task_list)