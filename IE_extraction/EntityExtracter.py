from IE_extraction.Basic_modules import Basic_modules
from InferenceNLI import InferenceNLIModule
from InferenceEntitytype import InferenceEntitytype
from owlready2 import get_ontology

class EntityExtracterModule(Basic_modules):
    def __init__(self, onto_path, pos_model_path="TweebankNLP/bertweet-tb2_ewt-pos-tagging"
                 , ner_model_path='Gladiator/microsoft-deberta-v3-large_ner_conll2003'):
        super().__init__()
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        self.pos_tokenizer = AutoTokenizer.from_pretrained(pos_model_path)
        self.model_list['pos_tagger'] = AutoModelForTokenClassification.from_pretrained(pos_model_path)
        self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
        self.model_list['ner'] = AutoModelForTokenClassification.from_pretrained(ner_model_path)
        self.nli_model = InferenceNLIModule()
        self.onto = get_ontology(onto_path).load()

    # def set_kr_ner_model(self):
    #     id2label_dict = self.model_list['ner'].config.id2label
    #     id2label_dict = {key:id2label_dict[key][-1] + id2label_dict[key][-2]+id2label_dict[key][:-2] for key in id2label_dict.keys() if key!=0}
    #     id2label_dict[0]='O'
    #     self.model_list['ner'].config.id2label = id2label_dict
    #     self.model_list['ner'].config.label2id = {v: k for k, v in id2label_dict.items()}

    def extract_entities(self, concated_dataset, use_neutral=False, threshold=0.8):
        print('extracting entities...')
        self.nli_model.load_model_on_device(self.device)
        from transformers import pipeline
        from transformers.pipelines.pt_utils import KeyDataset
        ner_extractor = pipeline('token-classification', model=self.model_list['ner'], tokenizer=self.ner_tokenizer,
                                 device = self.device, aggregation_strategy="simple")

        from tqdm.auto import tqdm
        propns_list = []
        for propns in tqdm(ner_extractor(KeyDataset(concated_dataset, 'translated_summary'), batch_size=32)):
            propns_list.append(propns)

        entity_decider = InferenceEntitytype()
        entity_decider.load_model_on_device(self.device)
        entity_decider.set_entity_class(self.onto)

        print('setting coarse-class entities')
        coarse_entities = []
        for propns in tqdm(propns_list):
            coarse_res = entity_decider.decide_coarse_entity_type(propns)
            coarse_entities.append(coarse_res)

        print('\nsetting fine-class entities')
        fine_entities =[]
        for propns in tqdm(coarse_entities):
            coarse_res = entity_decider.decide_fine_entity_type(propns)
            fine_entities.append(coarse_res)

        entity_decider.unload()
        self.nli_model.unload()
        return fine_entities
