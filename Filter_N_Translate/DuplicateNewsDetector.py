from .BasicModule import BasicModule
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import time
from datasets import Dataset

class DuplicateNewsDetectorModule(BasicModule):
    def erase_duplicated_single_row(self, grouped_dataset : Dataset):
        duplicated_idxs = []
        for i in range(len(grouped_dataset)):
            for j in range(i):
                if i==j : continue
                if grouped_dataset[i]['title'] == grouped_dataset[j]['title']:
                    duplicated_idxs.append(j)

        duplicated_idxs = list(set(duplicated_idxs))
        erased_dataset = grouped_dataset.filter(lambda example, idx: idx not in duplicated_idxs, with_indices=True)
        return erased_dataset

    def erase_similar_single_row(self, dup_erased_articles, threshold=0.35):
        if len(dup_erased_articles)==1: return dup_erased_articles, {}

        # Compute embeddings
        if self.lang=='KOR':
            embeddings = self.model.encode(dup_erased_articles['text'], convert_to_tensor=True)
        else:
            embeddings = self.model.encode(dup_erased_articles['title'], convert_to_tensor=True)

        from sklearn.cluster import DBSCAN
        embeddings_np = embeddings.cpu().numpy()
        dbscan = DBSCAN(eps=threshold, min_samples=1, metric='cosine')
        dbscan.fit(embeddings_np)

        # Create clusters
        from collections import defaultdict
        clusters = defaultdict(list)
        for idx, label in enumerate(dbscan.labels_):
            clusters[label].append(idx)

        processed_clusters=[]
        erase_target = []
        for key in clusters.keys():
            temp = clusters[key].copy()
            repre = temp.pop(0)
            if len(temp)>0:
                processed_clusters.append({'representative': [[dup_erased_articles['original_index'][repre]],dup_erased_articles['title'][repre]],
                                           'others': [[temp],dup_erased_articles.select(temp)['title']]})
                erase_target += temp
            else:
                processed_clusters.append({'representative' : [[],dup_erased_articles['title'][repre]], 'others' : []})


        sim_erased_articles = dup_erased_articles.filter(lambda row, idx: idx not in erase_target, with_indices=True)

        return sim_erased_articles, processed_clusters


    def erase_similar_and_duplicated(self, dataset:Dataset, filtering_thres=0.35):
        self.model = SentenceTransformer(self.model_path)

        print('\nErasing similar news...')
        start_time = time.time()

        datelist = list(set(dataset['datetime']))
        grouped_index_dict = {x:[] for x in datelist}
        for n, data in enumerate(dataset):
            grouped_index_dict[data['datetime']].append(n)

        dummy_features = {x:[] for x in dataset.features.copy()}
        res_dataset = Dataset.from_dict(dummy_features)

        print('==Deep erasing==')
        from datasets import concatenate_datasets, disable_progress_bar
        for key in tqdm(grouped_index_dict.keys()):
            disable_progress_bar()
            grouped_dataset = dataset.select(grouped_index_dict[key])
            if len(grouped_index_dict[key])<2:
                res_dataset = concatenate_datasets([res_dataset, grouped_dataset])
            else:
                dup_erased_article = self.erase_duplicated_single_row(grouped_dataset)
                sim_erased_articles, duplicated_clusters = self.erase_similar_single_row(dup_erased_article, threshold=filtering_thres)
                res_dataset = concatenate_datasets([res_dataset, sim_erased_articles])

        return self.save(res_dataset)