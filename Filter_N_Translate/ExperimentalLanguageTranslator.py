from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM
from optimum.bettertransformer import BetterTransformer
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from scipy.stats import entropy as calculate_entropy
# maybe KETI-AIR-Downstream/long-ke-t5-base-translation-aihub-ko2en is better. (longT5 based model)
from BasicModule import BasicModule

class LanguageTranslateModule(BasicModule):
    def __init__(self, num_workers, model_path, tokenizer_path):
        super().__init__(num_workers, model_path, tokenizer_path)
        self.config = PeftConfig.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, model_path)
        self.model = model.merge_and_unload()
        self.model = BetterTransformer.transform(model)

    def capture_uncertain(self, res_tensors, thres=3.5):
        res = []
        from torch.distributions import Categorical
        for idx in range(res_tensors.shape[1]):
            # logit -> prob.
            probs = res_tensors[:, idx, :].softmax(dim=-1)

            # find end index of summary
            try:
                sum_end = torch.where(res_tensors[:, idx, :].max(axis=1).indices == 2)[0][0]
            except IndexError:
                print('error')
                res.append([-1])
                continue
            # calculate entropies of each distribution
            entropies = Categorical(probs).entropy()[:sum_end]

            # find high entropy index
            suspect_index = torch.where(entropies >= thres)

            if len(suspect_index[0]) == 0:
                res.append([-1])
            else:
                suspect_indexes = suspect_index[0].tolist()
                temp = []
                for sus in suspect_indexes:
                    temp.append(sus)
                    # if [sus-1] in suspect_indexes or [sus+1] in suspect_indexes:
                    #     temp.append(sus)
                if len(temp) == 0:
                    res.append([-1])
                else:
                    res.append(temp)
            # TODO : </s> 처럼 문장이 끝나는 구간은 버리기
            # 핵심알고리즘 : span(연속된 토큰들) 의 엔트로피가 다 높은 구간을 포착하기
            # 그게 잘못된 고유명사일수 있을듯
        return res


    def deep_translate_batches_experiment(self, flatten_text, batch_size=48):
        from transformers import AutoTokenizer, DataCollatorWithPadding
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name_or_path, src_lang="kor_Hang")
        collator = DataCollatorWithPadding(tokenizer, padding=True)

        def tokenize_text(row):
            return tokenizer(row['text'], truncation=True, max_length=220)

        dataset = Dataset.from_dict({'text': flatten_text})
        print('\n==Tokenizing==')
        dataset_for_translate = dataset.map(tokenize_text, num_proc=self.num_workers, remove_columns=['text'])
        dataset_for_translate.set_format('torch')

        print('\n==Deep translation==')
        res = []
        exp_res = []
        dataloader = DataLoader(dataset_for_translate, batch_size=batch_size, num_workers=self.dataloader_workers,
                                collate_fn=collator)
        with torch.no_grad():
            for chunk in tqdm(dataloader):
                chunk = {k: v.to(self.device) for k, v in chunk.items()}
                # translated_tokens = self.model.generate(
                #     **chunk, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=250
                # )
                translated_tokens = self.model.generate(
                    **chunk, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=300
                    , return_dict_in_generate=True, output_scores=True)
                res += tokenizer.batch_decode(translated_tokens['sequences'], skip_special_tokens=True)
                temp = self.capture_uncertain(torch.stack(translated_tokens['scores'], axis=0))

                for n, captured_idxes in enumerate(temp):
                    if captured_idxes == [-1]:
                        exp_res.append(['pass'])
                    else:
                        captured_token = []
                        for idx in captured_idxes:
                            captured_token.append(tokenizer._convert_id_to_token(translated_tokens['sequences'][n, idx]))
                            if '<pad>' in captured_token:
                                print('error')
                        exp_res.append(captured_token)
        return res, exp_res


    def deep_translate_experiment(self, sentence_split_dataset: Dataset, grouped_index_dict, SPL=2, save_results=True,
                                  check_time=True):
        print('translating news...')
        s = time.time()
        if SPL == 1:
            from util.utils import flatten_and_batch_dict, return_batch_to_list
            # 각 문서들을 일정한 배치로 변형
            batch_hashmap, flatten_text = flatten_and_batch_dict(sentence_split_dataset['split_text'])

            # NLLB 번역
            mixed_result = self.deep_translate_batches(flatten_text, batch_size=64)

            # 배치 텍스트를 각 날짜로 그룹화
            result_list = return_batch_to_list(batch_hashmap, mixed_result)
            translated_dataset = sentence_split_dataset.map(lambda row, idx: {'translated_text': result_list[idx]},
                                                            with_indices=True)

        elif SPL == 0:
            mixed_result, exp_result = self.deep_translate_batches_experiment(sentence_split_dataset['summarized_text'],
                                                                              batch_size=8)
            translated_dataset = sentence_split_dataset.map(
                lambda row, idx: {'translated_summary': mixed_result[idx], 'error_captured': exp_result[idx]},
                with_indices=True)

        if save_results == True:
            translated_dataset = translated_dataset.select_columns(
                ['summarized_text', 'translated_summary', 'error_captured'])
            for key in grouped_index_dict.keys():
                if len(grouped_index_dict[key]) > 0:
                    save_target = translated_dataset.select(grouped_index_dict[key])
                    save_target.to_csv(f'{self.output_path}/{key}.csv')

                    error_captured_target = [idx for idx in grouped_index_dict[key] if exp_result[idx] != ['pass']]
                    if len(error_captured_target) != 0:
                        error_save_target = translated_dataset.select(error_captured_target)
                        error_save_target.to_csv(f'{self.output_path}/{key}_errors.csv')

        if check_time == True:
            print('time taken for translating news : ', time.time() - s)

        return translated_dataset, grouped_index_dict