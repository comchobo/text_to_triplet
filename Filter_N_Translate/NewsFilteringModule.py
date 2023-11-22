from datasets import Dataset
from .BasicModule import BasicModule
class NewsFilteringModule(BasicModule):
    def _deep_categorize(self, dataset, thres, batch_size):
        import torch
        from tqdm import tqdm
        import numpy as np
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        def tokenize_title_text(row):
            return tokenizer(row['title'], row['text'], padding = 'max_length', truncation=True, max_length=128)

        irrelevant_columns = [x for x in dataset.features.keys() if x not in ['index','input_ids','token_type_ids','attention_mask']]
        dataset_for_filter = dataset.map(tokenize_title_text, keep_in_memory=True)
        dataset_for_filter = dataset_for_filter.remove_columns(irrelevant_columns)
        dataset_for_filter.set_format('torch')

        print('\n==Deep filtering==')
        res = []
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        dataloader = DataLoader(dataset_for_filter, batch_size=batch_size, num_workers=self.dataloader_workers)
        for batch in tqdm(dataloader):
            with torch.no_grad():
                batch_index = batch.pop('index')
                batch = {k: v.to(self.device) for k, v in batch.items()}
                predicted = self.model(**batch)
            predicted_label = predicted.logits[:, 1]
            predicted_label = predicted_label.detach().cpu().numpy()
            predicted_label = np.argwhere(predicted_label >= thres).flatten()
            res+=batch_index[predicted_label].tolist()
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        return res

    from datasets import Dataset
    def deep_filter(self, concated_dataset : Dataset, filtering_thres = -0.4):
        import time
        start_time = time.time()
        idx_given_dataset = concated_dataset.add_column('index', [x for x in range(len(concated_dataset))])
        idx_given_dataset = idx_given_dataset.remove_columns('__index_level_0__')
        del concated_dataset

        categorized_idxs = self._deep_categorize(idx_given_dataset, thres=filtering_thres, batch_size=256)
        categorized_dataset = idx_given_dataset.select(categorized_idxs)
        del idx_given_dataset

        return self.save(categorized_dataset)
