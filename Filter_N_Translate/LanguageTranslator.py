from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM
from optimum.bettertransformer import BetterTransformer
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from .BasicModule import BasicModule

class LanguageTranslateModule(BasicModule):
    def __init__(self, num_workers, model_path, tokenizer_path):
        super().__init__(num_workers, model_path, tokenizer_path)
        self.config = PeftConfig.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, model_path)
        self.model = model.merge_and_unload()
        self.model = BetterTransformer.transform(model)

    def deep_translate_batches(self, flatten_text, batch_size = 48):
        from transformers import AutoTokenizer, DataCollatorWithPadding
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name_or_path, src_lang="kor_Hang")
        collator = DataCollatorWithPadding(tokenizer, padding=True)
        res = []
        def tokenize_text(row):
            return tokenizer(row['text'], truncation=True, max_length=220)

        dataset = Dataset.from_dict({'text':flatten_text})
        print('\n==Tokenizing==')
        dataset_for_translate = dataset.map(tokenize_text, num_proc=self.num_workers, remove_columns=['text'])
        dataset_for_translate.set_format('torch')

        print('\n==Deep translation==')
        dataloader = DataLoader(dataset_for_translate, batch_size=batch_size, num_workers=self.dataloader_workers, collate_fn=collator)
        with torch.no_grad():
            for chunk in tqdm(dataloader):
                chunk = {k: v.to(self.device) for k, v in chunk.items()}
                translated_tokens = self.model.generate(
                    **chunk, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=300
                )
                res += tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return res

    def deep_translate(self, sentence_split_dataset : Dataset, SPL=2, save_results=True, check_time=True):
        print('translating news...')
        self.model.to(self.device)
        if SPL==1:
            from util.utils import flatten_and_batch_dict, return_batch_to_list
            # 각 문서들을 일정한 배치로 변형
            batch_hashmap, flatten_text = flatten_and_batch_dict(sentence_split_dataset['split_text'])

            # NLLB 번역
            mixed_result = self.deep_translate_batches(flatten_text, batch_size=48)

            # 배치 텍스트를 각 날짜로 그룹화
            result_list = return_batch_to_list(batch_hashmap, mixed_result)
            translated_dataset = sentence_split_dataset.map(lambda row, idx: {'translated_text':result_list[idx]}, with_indices=True)
        elif SPL==0:
            mixed_result = self.deep_translate_batches(sentence_split_dataset['summarized_text'], batch_size=32)
            translated_dataset = sentence_split_dataset.map(
                lambda row, idx: {'translated_summary': mixed_result[idx]}, with_indices=True)
        elif SPL==2:
            mixed_result = self.deep_translate_batches(sentence_split_dataset['text'], batch_size=32)
            translated_dataset = sentence_split_dataset.map(lambda row, idx: {'translated_summary': mixed_result[idx]}, with_indices=True)
        else:
            print('Please select SPL mode')
            exit()

        return self.save(translated_dataset)

