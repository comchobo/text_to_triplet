from Filter_N_Translate.Preprocessor import DeepPreprocessorModule
from Filter_N_Translate.NewsFilteringModule import NewsFilteringModule
from Filter_N_Translate.SentenceSplitter import SentenceSplitterModule
from Filter_N_Translate.LanguageTranslator import LanguageTranslateModule
from Filter_N_Translate.DuplicateNewsDetector import DuplicateNewsDetectorModule
# from Filter_N_Translate.EntityLinkedLanguageTranslator import EntityLinkedLanguageTranslateModule

import os, json, gc, sys, random
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm


class ModuleController:
    def __init__(self, input_data, args):
        self.data = input_data
        self.args = args

    def preprocess(self):
        args = self.args
        preprocessor = DeepPreprocessorModule(num_workers=args.num_worker, model_path=args.preprocessing_model_path
                                              , tokenizer_path = args.preprocessing_tokenizer_path, lang_mode=args.lang_mode
                                              , save_results=True, check_time=True)
        preprocessor.set_output_path(f'{args.output_path}/1')
        preprocessor.set_device(args.device)
        self.data = preprocessor.deep_preprocess(self.data)

    def filter(self):
        args = self.args
        newsfilterer = NewsFilteringModule(num_workers=args.num_worker, model_path=args.filtering_model_path
                                           , tokenizer_path = args.filtering_tokenizer_path, lang_mode=args.lang_mode
                                           , save_results=True, check_time=True)
        newsfilterer.set_output_path(f'{args.output_path}/1_')
        newsfilterer.set_device(args.device)
        self.data = newsfilterer.deep_filter(self.data, filtering_thres=-0.4)


    def erase_similar(self):
        args = self.args
        duperaser = DuplicateNewsDetectorModule(num_workers=args.num_worker, model_path=args.eraser_model_path
                                                , tokenizer_path=args.eraser_model_path, lang_mode=args.lang_mode
                                                , save_results=True, check_time=True)
        duperaser.set_output_path(f'{args.output_path}/1_2_')
        duperaser.set_device(args.device)
        self.data = duperaser.erase_similar_and_duplicated(self.data, filtering_thres=0.35)


    def split_sentence(self):
        args = self.args
        splitter = SentenceSplitterModule(num_workers=args.num_worker, model_path=None
                                          , tokenizer_path = None, lang_mode=args.lang_mode)
        splitter.set_output_path(f'{args.output_path}/1_2_3(SPL)')
        self.data = splitter.split_sentence(self.data)


    def translate(self):
        args = self.args
        translator = LanguageTranslateModule(num_workers=args.num_worker, model_path=args.translation_model_path,
                                              tokenizer_path = args.translation_model_path)
        translator.set_output_path(f'{args.output_path}/1_2_3(SPL)_4')
        translator.set_device(args.device)
        self.data = translator.deep_translate(self.data, SPL=1)

    def output_data(self):
        return self.data

# todo
# def entity_linked_translate(dataset, args):
#     torch.cuda.empty_cache()
#     translator = EntityLinkedLanguageTranslateModule(num_workers=args.num_worker, model_path=args.translation_model_path,
#                                           tokenizer_path = args.translation_model_path)
#     translator.set_output_path(f'{args.output_path}/1_2_3(SPL)_4(EL)')
#     translator.set_device(args.device)
#     translated_dataset = translator.deep_translate(dataset, SPL=1)
#     return translated_dataset


def main(args, task_list):
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
            concated_dataset = load_dataset("json", data_files=files_of_years)['train']

        parser = argparse.ArgumentParser()
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
        module_controller = ModuleController(concated_dataset, args_for_modules)

        task_methods = {
            'preprocess': module_controller.preprocess,
            'filter': module_controller.filter,
            'split_sentence': module_controller.split_sentence,
            'erase_similar': module_controller.erase_similar,
            'translate': module_controller.translate,
        }

        for task_name in task_list:
            task_method = task_methods.get(task_name)
            task_method()
        result_data = module_controller.output_data()
        result_data.save(args.output_path)

        del concated_dataset
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__' :
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='./input')
    parser.add_argument('--output_path', default='./output')

    parser.add_argument('--year_range',default='2017-2024')
    parser.add_argument('--debug_mode', default=False, help='This will set the dataset size and proc_num')
    parser.add_argument('--lang_mode', default='KOR')
    parser.add_argument('--tasks', nargs='+', help='List of tasks to run', required=True)

    args = parser.parse_args()
    # args.task를 직접 정의할 수 있습니다. ['함수명1', '함수명2', ...]
    main(args, args.tasks)