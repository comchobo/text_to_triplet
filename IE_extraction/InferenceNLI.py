from transformers import AutoTokenizer, AutoModelForSequenceClassification
from IE_extraction.Basic_modules import Basic_modules

class InferenceNLIModule(Basic_modules):
    def __init__(self, model_path='sileod/deberta-v3-base-tasksource-nli'):
        super().__init__()
        self.model_list['nli'] = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def run_nli(self, premises, hypotheses, max_length=512, argmax=True, use_neutral=True):
        if type(premises) == str:
            premises = [premises for _ in range(len(hypotheses))]
        assert len(premises) == len(hypotheses)

        inputs = self.tokenizer(premises, hypotheses, padding=True, truncation='only_first', max_length=max_length,
                           return_tensors='pt').to(self.device)
        outputs = self.model_list['nli'](**inputs)

        if use_neutral:
            probs = outputs.logits.softmax(dim=-1)[:, 0]
        else:
            probs = outputs.logits[:, [0, 2]].softmax(dim=0)[:, 0]

        if argmax:
            pred = probs.argmax().item()
            prob = probs[pred].item()
            return pred, prob
        else:
            probs = probs.tolist()
            preds = [(pred, prob) for pred, prob in enumerate(probs)]
            return preds