from .BasicModule import BasicModule
from transformers import BigBirdTokenizerFast
import re
from transformers import BigBirdForTokenClassification, DataCollatorForTokenClassification
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from transformers import logging
# from Filter_N_Translate.RulebasedPreprocessor import Preprocessor # deprecated

# pre-process before tokenizing. Those characters are recognized as 'UNK'.
class CustomBigBirdTokenizer(BigBirdTokenizerFast):
    def _encode_plus(self, text, **kwargs):
        text = re.sub('―', '-', text)
        text = re.sub('？', '-', text)
        text = re.sub('[“”]', '"', text)
        text = re.sub('[’‘]', "'", text)
        text = re.sub('\\\\n', " ", text)
        text = re.sub('\n', " ", text)
        text = re.sub(r'[\u4e00-\u9fff]', '', text)
        return super()._encode_plus(text, **kwargs)


class DeepPreprocessorModule(BasicModule):

    def _deep_preprocess(self, text_list, batch_size):
        from tqdm import tqdm
        self.model = BigBirdForTokenClassification.from_pretrained(self.model_path)
        if self.lang=='KOR':
            self.tokenizer = CustomBigBirdTokenizer.from_pretrained(self.tokenizer_path)
        elif self.lang=='ENG':
            self.tokenizer = CustomBigBirdTokenizer.from_pretrained(self.tokenizer_path)

        self.model.to(self.device)
        collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        dataset = Dataset.from_dict({'text':text_list})
        def tokenize_text(row):
            logging.set_verbosity_error()
            return self.tokenizer(row['text'], padding = 'longest', truncation=True, max_length=4096,
                                                pad_to_multiple_of=1024)

        dataset_for_preprocess = dataset.map(tokenize_text, num_proc=self.num_workers, keep_in_memory=True)
        dataset_for_preprocess.set_format('torch')
        dataset_for_preprocess = dataset_for_preprocess.remove_columns(['text'])

        print('\n==Deep Preprocessing==')
        temp_labels = []
        temp_attentions=[]
        temp_ids = []

        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        dataloader = DataLoader(dataset_for_preprocess, batch_size=batch_size, num_workers=self.dataloader_workers
                                , collate_fn=collator)
        for batch in tqdm(dataloader):
            with torch.no_grad():
                batch = {k: v.to(self.device) for k, v in batch.items()}
                predicted = self.model(**batch)
                temp_labels.append(torch.argmax(predicted.logits, axis=2))
                temp_attentions.append(batch['attention_mask'])
                temp_ids.append(batch['input_ids'])

        # labeled **words** are not tokens. so following operation will manage to match the gap
        padder_attentions = [torch.zeros(batch_size, 4096-x.shape[1]) for x in temp_attentions[:-1]]
        padder_attentions.append(torch.zeros(temp_attentions[-1].shape[0], 4096 - temp_attentions[-1].shape[1]))
        padded_attentions = [torch.cat((temp_attentions[idx].cpu(), padder_attentions[idx]), 1) for idx in
                             range(len(padder_attentions))]
        padded_labels = [torch.cat((temp_labels[idx].cpu(), padder_attentions[idx]), 1) for idx in
                             range(len(padder_attentions))]
        padded_ids = [torch.cat((temp_ids[idx].cpu(), padder_attentions[idx]), 1) for idx in
                             range(len(padder_attentions))]

        predicted_labels = torch.cat(padded_labels, dim=0)
        batch_lengths = torch.cat(padded_attentions, dim=0)
        batch_ids = torch.cat(padded_ids, dim=0)

        res_ids = []
        for idx in tqdm(range(len(predicted_labels))):
            batch_length = torch.argwhere(batch_lengths[idx,:]==1).shape[0]
            predicted_label = predicted_labels[idx,:batch_length]
            if self.lang=='ENG':
                predicted_label = torch.argwhere(predicted_label != 1).flatten()
            else:
                predicted_label = torch.argwhere(predicted_label!=0).flatten()
            res_ids.append([int(batch_ids[idx,x].item()) for x in predicted_label])
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        from tqdm import tqdm
        res = self.tokenizer.batch_decode(tqdm(res_ids), skip_special_tokens=True)
        res = [x[2:] if x.startswith('##') else x for x in res]
        return res

    from datasets import Dataset
    def deep_preprocess(self, concated_dataset : Dataset):
        # preprocessor = Preprocessor()
        # print('checking the errors')
        # concated_dataset = concated_dataset.filter(lambda row : preprocessor.check_error(row) == 0
        #                                            , num_proc=self.num_workers)

        preprocessed_text = self._deep_preprocess(concated_dataset['text'], batch_size=6)
        concated_dataset = concated_dataset.remove_columns(['text'])
        concated_dataset = concated_dataset.add_column('text', preprocessed_text)
        del preprocessed_text

        return self.save(concated_dataset)
