def change_words(s, words, indices):
    if len(words) != len(indices):
        raise ValueError("Number of words and number of indices must be the same")

    # Calculate the offset for each replacement
    offset = 0

    for i, (start, end) in enumerate(indices):
        # Calculate the adjusted indices with the offset
        adj_start = start + offset
        adj_end = end + offset

        # Replace the substring with the target word
        s = s[:adj_start] + words[i] + s[adj_end:]

        # Update the offset
        offset += len(words[i]) - (end - start)

    return s

class Coreferrencer:
    def __init__(self, num_workers):
        self.num_workers = num_workers

    def set_output_path(self,output_path):
        self.output_path = output_path

    def load_model_on_device(self, device):
        self.device = device

    def unload(self):
        del self.model

    def load_dataset_from_path(self, filepath):
        from datasets import Dataset
        import pandas as pd
        from tqdm import tqdm
        import os
        concated_news = pd.concat(map(pd.read_csv,tqdm(os.listdir(filepath))))
        concated_dataset = Dataset.from_pandas(concated_news)
        return concated_dataset

    # pending.
    # def load_entity_vocab(self, path):
    #     self.vocab = {'name':{'id':'','subclass':'','type':''}}
    #
    # def find_from_entity_vocab(self, element_list):
    #     for element in element_list:
    #         if element

    def set_representative(self, strings:[]):
        import re
        strings = list(set(strings))

        temp_st = strings[:]
        for _ in range(len(strings)):
            candidate = max(temp_st, key=lambda s: len(s))
            if re.match('[A-Z]', candidate) is not None:
                if candidate[-2:]=="'s":
                    return candidate[:-2]
                else: return candidate
            temp_st.remove(candidate)
        return None

    # pending : TODO
    # 만약 임베딩 vocab안에서 찾았을 때는, 해당 단어로 모두 치환한다.
    def post_processing_corefs(self, pred):
        clusters_idxes = pred.get_clusters(as_strings=False),
        clusters_strings = pred.get_clusters(as_strings=True)

        representatives = [self.set_representative(x) for n, x in enumerate(clusters_strings)]

        # 제일 길고 대문자가 있는 것을 representative로 만들었다.
        for n, rep in enumerate(representatives):
            if rep is None : continue


                # 클러스터에서 동사는 제외한다
                # 제일 긴 element를 key로 만든다?

            key = max(strings, key=lambda s: len(s)) # 제일 긴 element를 key로 만든다?


        # value를 string 위치로 잡고 이를 key로 만든다. -> [string start:end] 를 리스트단위로 받았을 때 한꺼번에 바꾸는 함수
        # change_words()

    from datasets import Dataset
    def resolve_coref(self, concated_dataset : Dataset, grouped_index_dict, check_time=True):
        import time
        print('converting pronouns...')
        s = time.time()

        from fastcoref import LingMessCoref
        from tqdm import tqdm
        import logging
        preds = []
        self.model = LingMessCoref(device=self.device, nlp='en_core_web_trf', enable_progress_bar=False)
        logger = logging.getLogger()
        logger.disabled = True

        for key in tqdm(grouped_index_dict.keys()):
            oneday_articles = concated_dataset.select(grouped_index_dict[key])
            joined_paragraph = [" ".join(x) for x in oneday_articles['translated_text']]
            preds.append(self.model.predict(joined_paragraph))

            for pred in preds:
                post_processed_pred = post_processing_corefs(pred)

            translated_dataset = sentence_split_dataset.map(lambda row, idx: {'translated_text': result_list[idx]},
                                                            with_indices=True)

        for p in preds:
            data = {'text': p.text,
                     'clusters': p.get_clusters(as_strings=False),
                     'clusters_strings': p.get_clusters(as_strings=True)}
        # data = [{'text': p.text,
        #          'clusters': p.get_clusters(as_strings=False),
        #          'clusters_strings': p.get_clusters(as_strings=True)}
        #         for p in preds]

        if check_time == True:
            print('time taken for co-referrencing news : ',time.time()-s)

        return concated_dataset
