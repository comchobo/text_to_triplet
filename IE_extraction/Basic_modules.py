class Basic_modules():
    def __init__(self):
        self.model_list = {}

    def set_output_path(self,output_path):
        self.output_path = output_path

    def load_model_on_device(self, device):
        self.device = device
        for key in self.model_list.keys():
            self.model_list[key].to(device)

    def unload(self):
        model_names = list(self.model_list.keys()).copy()
        for key in model_names:
            del self.model_list[key]

    def load_dataset_from_path(self, filepath):
        from datasets import Dataset
        import pandas as pd
        from tqdm import tqdm
        import os
        concated_news = pd.concat(map(pd.read_csv,tqdm(os.listdir(filepath))))
        concated_dataset = Dataset.from_pandas(concated_news)
        return concated_dataset