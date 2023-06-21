from datasets import Dataset, load_dataset

DATANAME = {'1.0.0': 'WENGSYX/Lingo-dataset-v1'}
DATA_LETEST = '1.0.0'


class LingoDataset():
    def __init__(self, version='letest'):
        assert version in ['letest', '1.0.0']

        if version == 'letest':
            version = DATA_LETEST
        self.origin_dataset = load_dataset(DATANAME[version])

        self.dataset = self.origin_dataset['train']
        self.version = version

    def get_list(self):
        return self.dataset.to_list()

    def get_dict(self):
        return self.dataset.to_dict()

    def set_model_name(self, name):
        datasets_list = self.get_list()

        def gen():
            for d in datasets_list:
                conver = []
                for con in d['conversations']:
                    conver.append(con.replace('[MODEL NAME]', name))
                d['conversations'] = conver
                yield d

        self.dataset = Dataset.from_generator(gen)

    def add_sample(self, sample):
        assert type(sample) == list
        self.dataset = self.dataset.add_item(
            {'conversations': sample, 'source': 'self_add_sample', 'version': self.version + '.1'})

    def push_to_hub(self, repo_id):
        self.dataset.push_to_hub(repo_id)
