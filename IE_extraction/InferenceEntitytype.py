from IE_extraction.Basic_modules import Basic_modules

class InferenceEntitytype(Basic_modules):
    def __init__(self, model_path='sileod/deberta-v3-base-tasksource-nli', max_length=64):
        super().__init__()
        self.model_path = model_path
        self.max_length=max_length
        self.coarse_NER_dict = 'blinded'

    def set_entity_class(self, onto):
        coarse_entity_class = list(onto.Entity.subclasses())
        self.coarse_entity_class_dict = {x.name:x for x in coarse_entity_class}
        flatten_entity_class = []
        for x in coarse_entity_class:
            flatten_entity_class += x.subclasses()
        self.fine_entity_class = {x.name:x for x in flatten_entity_class}

    def load_model_on_device(self, device):
        from transformers import pipeline
        self.pipe = pipeline(model=self.model_path, device=device,
                         max_length=self.max_length)

    def post_process_entities(self, entity_dicts):
        entities_ready = []
        entities_MISC = []
        temp_entity_names = []

        for entity_dict in entity_dicts:
            temp_dict= {}
            if entity_dict['word'] not in temp_entity_names:
                temp_dict['word'] = entity_dict['word']
                temp_entity_names.append(entity_dict['word'])
                if self.coarse_NER_dict.get(entity_dict['entity_group'], False) is not False:
                    temp_dict['type'] = self.coarse_NER_dict[entity_dict['entity_group']]
                    entities_ready.append(temp_dict)
                else:
                    temp_dict['type'] = 'none'
                    entities_MISC.append(temp_dict)

        return entities_ready, entities_MISC

    def decide_coarse_entity_type(self, entity_dicts):
        entities_ready, entities_MISC = self.post_process_entities(entity_dicts)
        res = self.pipe([x['word'] for x in entities_MISC], candidate_labels=list(self.coarse_entity_class_dict.keys()))
        MISC_types = [x['labels'][0] for x in res]
        entities_MISC = [{'word':entities_MISC[i]['word'], 'type':MISC_types[i]} for i in range(len(MISC_types))]
        return entities_ready + entities_MISC

    def decide_fine_entity_type(self, entity_dicts):
        entity_dicts_misc = [x for x in entity_dicts if x['type']!='Person']
        entity_dicts_person = [{'text':x['word'], 'type':self.coarse_entity_class_dict[x['type']]}
                               for x in entity_dicts if x['type']=='Person']

        res = self.pipe([x['word'] for x in entity_dicts_misc], candidate_labels=list(self.fine_entity_class.keys()))
        res_types = [x['labels'][0] for x in res]
        fine_typed_entities = [{'text':entity_dicts_misc[i]['word'], 'type':self.fine_entity_class[res_types[i]]}
                               for i in range(len(res_types))]

        return fine_typed_entities + entity_dicts_person