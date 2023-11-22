from IE_extraction.Basic_modules import Basic_modules
from InferenceNLI import InferenceNLIModule
from owlready2 import get_ontology

class RelationMatcherModule(Basic_modules):
    def __init__(self, onto_path, threshold=0.8):
        super().__init__()
        self.onto = get_ontology(onto_path).load()
        self.nli_model = InferenceNLIModule()
        self.threshold = threshold

    def get_candidate_relations(self, sbj_type, obj_type):
        relations = []
        for rel in self.onto.object_properties():
            sbj_ok = sbj_type.ancestors() & set(rel.domain)
            obj_ok = obj_type.ancestors() & set(rel.range)
            if sbj_ok and obj_ok:
                relations.append(rel)
        return relations

    def extract_relations(self, premise, entities):
        triplets = []
        for sbj in entities:
            for obj in entities:
                if sbj == obj: continue

                relations = self.get_candidate_relations(sbj['type'], obj['type'])
                if len(relations) == 0: continue

                hypotheses, relations_ = [], []
                for r in relations:
                    for l in r.comment:
                        hypotheses.append(l.format(sbj=sbj['text'], obj=obj['text']))
                        relations_.append(r.name)
                if len(hypotheses) == 0: continue

                # preds = run_nli(model, tokenizer, premise, relation_hypothesis, argmax=False)
                # for pred, prob in preds:
                #     if prob > threshold:
                #         triplets.append((sbj, relations[pred], obj))

                pred, prob = self.nli_model.run_nli(premise, hypotheses, argmax=True)
                if prob > self.threshold:
                    triplets.append(({'text':sbj['text'],'type':sbj['type'].name},
                                     relations_[pred],
                                     {'text':obj['text'],'type':obj['type'].name}))
        return triplets

    def extract_relations_batch(self, ts_dataset, entities, save_results=True):
        self.nli_model.load_model_on_device(self.device)
        from tqdm import tqdm
        res_rel = []
        print('finding relations...')
        for i in tqdm(range(len(ts_dataset))):
            triplets = self.extract_relations(ts_dataset['translated_summary'][i], entities[i])
            res_rel.append(triplets)

        entities_to_save = []
        for entities_per_row in entities:
            temp = []
            for entity in entities_per_row:
                temp.append({'text':entity['text'],'type':entity['type'].name})
            entities_to_save.append(temp)

        triplets_from_EngSum= [{'original_title':ts_dataset['title'][i],
                                'original_text':ts_dataset['text'][i], 'translated_summary':ts_dataset['translated_summary'][i],
                                'found_entities': entities_to_save[i], 'found_relations': res_rel[i]}
                               for i in range(len(ts_dataset))]

        datelist = list(set(ts_dataset['datetime']))
        grouped_index_dict = {x:[] for x in datelist}
        for n, data in enumerate(ts_dataset):
            grouped_index_dict[data['datetime']].append(n)

        from datasets import disable_progress_bar
        import json
        disable_progress_bar()
        if save_results==True:
            for key in grouped_index_dict.keys():
                save_target = []
                for idx in grouped_index_dict[key]:
                    save_target.append(triplets_from_EngSum[idx])

                with open(f"{self.output_path}/{key}.jsonl", encoding="utf-8", mode="w") as file:
                    for i in save_target: file.write(json.dumps(i, ensure_ascii=False) + "\n")

        self.nli_model.unload()
        return triplets_from_EngSum