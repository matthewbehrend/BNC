import os
from mxnet.gluon.data import DataLoader

class DatasetGroup(object):

    def __init__(self, name, path=None, download=True):
        self.name = name
        self.train = None
        self.test  = None
        if path is None:
            path = os.path.join(os.getcwd(), 'data')
        self.path = path
        if download:
            self.download()

    def makeDomainDatasetLoader(self, array_dataset_train, array_dataset_test, batch_size):
            self.train = DataLoader(array_dataset_train, batch_size, shuffle=True,  last_batch='keep')
            self.test  = DataLoader(array_dataset_test,  batch_size, shuffle=False, last_batch='keep')
    
    def get_path(self, *args):
        return os.path.join(self.path, self.name, *args)

    def download(self):
        """Download the dataset(s).

        This method only performs the download if necessary. If the dataset
        already resides on disk, it is a no-op.
        """
        pass
    
    def reSample(self):
        pass


datasets = {}


def register_dataset(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def get_dataset(name, *args, **kwargs):
    return datasets[name](*args, **kwargs)
