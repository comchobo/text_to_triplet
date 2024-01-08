import os
from functools import wraps
import timeit

class BasicModule:
    def __init__(self, num_workers, model_path, tokenizer_path, lang_mode = 'ENG', check_time=True, save_results=True):
        self.num_workers = num_workers
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.lang = lang_mode
        self.check_time = check_time
        self.save_results = save_results
        self.decorator = timeit

        if num_workers>2 : self.dataloader_workers=2
        else: self.dataloader_workers = 0

    def set_output_path(self,output_path):
        if os.path.isdir(output_path) is False:
            os.mkdir(output_path)
        self.output_path = output_path

    def set_device(self, device):
        self.device = device

    def load_dataset_from_path(self, filepath):
        from datasets import Dataset
        import pandas as pd
        from tqdm import tqdm
        import os
        concated_news = pd.concat(map(pd.read_csv,tqdm(os.listdir(filepath))))
        concated_dataset = Dataset.from_pandas(concated_news)
        return concated_dataset

    def save(self, dataset):
        from datasets import disable_progress_bar, enable_progress_bar

        datelist = list(set(dataset['datetime']))
        grouped_index_dict = {x:[] for x in datelist}
        for n, data in enumerate(dataset):
            grouped_index_dict[data['datetime']].append(n)

        if self.save_results:
            disable_progress_bar()
            for key in grouped_index_dict.keys():
                save_target = dataset.select(grouped_index_dict[key])
                save_target.to_json(f'{self.output_path}/{key}.json', force_ascii=False, indent=4)
            enable_progress_bar()

        return dataset

    def conditional_timeit(self, dec):
        def decorator(func):
            if not self.check_time:
                return func
            return dec(func)

        return decorator
