import os
from dataset_loading.officehome import OfficeHomeArt, OfficeHomeClipart, OfficeHomeProduct, OfficeHomeReal
from dataset_loading.dataset import DatasetGroup
import numpy as np
from mxnet import gluon
from mxnet.gluon.data import ArrayDataset

class OfficeHomeDatasets(object):
    
    def __init__(self, useResNetFeatures=True, asdataloader=True):
        self.k_classes = 65
        self.useFeatures = useResNetFeatures
        print('OfficeHome Dataset. classes: ', self.k_classes)

        if(self.useFeatures):
            self.fn_cache = 'data_cache/officehomefeatures.npz'
        else:
            self.fn_cache = 'data_cache/officehome.npz'
        
        if(not os.path.exists( self.fn_cache )):
            self.readAndCacheData()

        self.load(asdataloader)

    def readAndCacheData(self):
        print('Loading data...')

        net = None
        if(self.useFeatures):
            net = getResNetFeatureExtractor()
        
        art        = OfficeHomeArt(extractor=net)
        clipart    = OfficeHomeClipart(extractor=net)
        product    = OfficeHomeProduct(extractor=net)
        real       = OfficeHomeReal(extractor=net)
        np.savez_compressed(self.fn_cache,
                            art.train._data[0],     art.train._data[1],     art.test._data[0],      art.test._data[1],
                            clipart.train._data[0], clipart.train._data[1], clipart.test._data[0],  clipart.test._data[1],
                            product.train._data[0], product.train._data[1], product.test._data[0],  product.test._data[1],
                            real.train._data[0],    real.train._data[1],    real.test._data[0],     real.test._data[1]
                            )

    def load(self, asdataloader):
        print('Loading data from cache')
        dat = np.load(self.fn_cache)
        batch_size = 256
        i = 0
        self.art        = _addDataset('art', dat, i, batch_size, asdataloader)
        i += 4
        self.clipart    = _addDataset('clipart', dat, i, batch_size, asdataloader)
        i += 4
        self.product    = _addDataset('product', dat, i, batch_size, asdataloader)
        i += 4
        self.real       = _addDataset('real', dat, i, batch_size, asdataloader)

        self.domains = {'art':self.art, 'clipart':self.clipart, 'product':self.product, 'real':self.real}
       
 
def getResNetFeatureExtractor():
    net = gluon.model_zoo.vision.resnet50_v1(pretrained=True).features
    return net

def _addDataset(name, dat, idx, batch_size, asdataloader):
    fl = dat.files
    train = ArrayDataset(dat[fl[idx]],   dat[fl[idx+1]])
    test  = ArrayDataset(dat[fl[idx+2]], dat[fl[idx+3]])

    dat_set = DatasetGroup( name )
    if(asdataloader):
        dat_set.makeDomainDatasetLoader(train, test, batch_size)
    else:
        dat_set.train = train
        dat_set.test = test
        
    return dat_set



