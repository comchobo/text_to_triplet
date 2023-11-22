class Summarize():
    def __init__(self, num_workers, model_name):
        from transformers import EncoderDecoderModel
        self.model_name = model_name
        self.model = EncoderDecoderModel.from_pretrained(model_name)
        self.num_workers = num_workers
        self.dataloader_workers = 0
        if num_workers>2 : self.dataloader_workers=2

    def load_model_on_device(self, device):
        self.model.to(device)
        self.device = device

    def set_output_path(self,output_path):
        self.output_path = output_path

    def unload(self):
        del self.model

    def summarize_batches(self, flatten_text, batch_size):
        from transformers import AutoTokenizer, DataCollatorWithPadding
        from datasets import Dataset
        from torch.utils.data import DataLoader
        import torch
        from tqdm import tqdm
        import time
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # collator = DataCollatorWithPadding(tokenizer, padding=True)
        res = []

        def tokenize_text(row):
            return tokenizer(row['text'], truncation=True, max_length=512, padding='max_length')

        dataset = Dataset.from_dict({'text':flatten_text})
        print('\n==Tokenizing==')
        dataset_for_summarize = dataset.map(tokenize_text, num_proc=self.num_workers, remove_columns=['text'], keep_in_memory=True)
        dataset_for_summarize = dataset_for_summarize.remove_columns(['token_type_ids'])
        dataset_for_summarize.set_format('torch')

        print('\n==Deep summarization==')
        dataloader = DataLoader(dataset_for_summarize, batch_size=batch_size, num_workers=self.dataloader_workers)
        with torch.no_grad():
            for chunk in tqdm(dataloader):
                chunk = {k: v.to(self.device) for k, v in chunk.items()}
                outputs = self.model.generate(
                    **chunk,
                    min_length=10,
                    max_length=250
                )
                res += tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return res

    def summarize(self, concated_news, grouped_index_dict, save_results=True, check_time=True):
        import time
        print('summarizing news...')
        s = time.time()

        # summarize
        summarized = self.summarize_batches(concated_news['text'], batch_size=16)
        summaried_dataset = concated_news.map(lambda row, idx: {'summarized_text':summarized[idx]}, with_indices=True)

        if save_results == True:
            for key in grouped_index_dict.keys():
                save_target = summaried_dataset.select(grouped_index_dict[key])
                save_target.to_csv(f'{self.output_path}/{key}.csv')

        if check_time==True:
            print('time taken for summarizing news : ',time.time()-s)

        return summaried_dataset, grouped_index_dict