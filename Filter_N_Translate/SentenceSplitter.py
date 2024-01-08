from .BasicModule import BasicModule
from kiwipiepy import Kiwi


class SentenceSplitterModule(BasicModule):
    def split_sentence(self, dup_erased_dataset):
        import time
        print('\nsentence splitting news...')
        temp = []
        kiwi = Kiwi()

        dup_erased_dataset = dup_erased_dataset.filter(lambda row: isinstance(row["text"], str) is True)
        dataset_SentenceObj = kiwi.split_into_sents(dup_erased_dataset['text'])
        for sentences in dataset_SentenceObj:
            try:
                temp.append([x.text for x in sentences])
            except ValueError:
                print('error occurred!')
                temp.append(['error'])

        sentence_split_dataset = dup_erased_dataset.add_column('split_text', temp)

        return self.save(sentence_split_dataset)
