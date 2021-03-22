import os
import numpy as np
from mxnet.gluon.data import ArrayDataset
from mxnet import nd
import mxnet as mx

from dataset_loading.dataset import DatasetGroup, register_dataset


@register_dataset('officehome')
class OfficeHome(DatasetGroup):

    base_url = ''

    data_domains = {
            'Art': 'Art',
            'Clipart': 'Clipart',
            'Product': 'Product',
            'Real World': 'Real World'
            }

    def __init__(self, subtype, path=None, shuffle=True, extractor=None):
        DatasetGroup.__init__(self, 'officehome', path=path)
        
        self.final_image_shape = (3, 224, 224)
        self.extractor = extractor
        self.subtype = subtype # the domain
        self.train_on_extra = False  # disabled
        self.label_shape = ()
        self.shuffle = shuffle
        self._load_datasets()

    def download(self):
        data_dir = self.get_path()
        if not os.path.exists(data_dir):
            print("Please download the OfficeHome dataset from https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view")
            print('Dataset homepage: https://www.hemanthdv.org/officeHomeDataset.html')
            print("Extract contents of zip file to ./data/officehome")
            print("subfolders of ./data/officehome should be Art, Clipart, Product, Real World")

    def _load_datasets(self):
        
        # load images for one of the four domains in OfficeHome
        domain_path = self.get_path(self.subtype)
        print('Loading ' + domain_path)
        
        # get the class folders
        _, dirnames, _ = next(os.walk(domain_path, (None, None, [])))
        
        # class index/name dictionaries
        self.class_to_index = dict(zip(dirnames, range(len(dirnames))))
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        print(self.index_to_class)
            
        all_images = None
        all_labels = np.zeros((0,), dtype='float32')
        
        # Read image files in the domain
        for label in self.index_to_class.keys():
            class_path = os.path.join(domain_path, self.index_to_class[label])
            _, _, filenames = next(os.walk(class_path, (None, None, [])))

            # initialize temporary variables
            new_labels = label * np.ones((len(filenames),), dtype='int32')
            new_images = np.zeros((len(filenames),) + self.final_image_shape, dtype='float32')

            print('reading class ', label)
            
            # Read images of the current class label
            for i, fn in enumerate(filenames):
                image_path = os.path.join(class_path, fn)
                image = mx.image.imread(image_path) # RGB image data
                image = self.transform(image)
                new_images[i, :, :, :] = np.moveaxis(image, [3], [1]) # rotate color axis 'iyxc->icyx'

            print('images size', new_images.shape)
            
            # Extract featutes, such as ResNet-50, if an extractor network was given
            if(self.extractor is not None):
                print('extracting features')
                new_images = self.extractor(nd.array(new_images)).asnumpy()
                
            if(all_images is not None):
                all_images = np.vstack((all_images, new_images))
            else:
                all_images = new_images      
            all_labels = np.concatenate((all_labels, new_labels))

            print('all images', all_images.shape)
            print('all labels', all_labels.shape)

        # Note: OfficeHome is a domain adaptation dataset and has no train/test split within a domain
        self.train = ArrayDataset(all_images, all_labels)
        self.test  = ArrayDataset(all_images, all_labels)

    def resampleSplit(self):
        pass

    def transform(self, img):
        w = np.max(img.shape[0:2]) # long edge size of image
        img, _ = mx.image.center_crop(img, (w, w), interp=2) # expand canvas to sqaure image
        
        s = self.final_image_shape[1:3]
        img = mx.image.imresize(img, s[0], s[1])
        img = img.astype(dtype='float32').asnumpy() / 255
        img = np.reshape(img, (1,) + img.shape)
        return img


class OfficeHomeArt(OfficeHome):
    def __init__(self, **kwargs):
        super().__init__(subtype='Art', **kwargs)


class OfficeHomeClipart(OfficeHome):
    def __init__(self, **kwargs):
        super().__init__(subtype='Clipart', **kwargs)


class OfficeHomeProduct(OfficeHome):
    def __init__(self, **kwargs):
        super().__init__(subtype='Product', **kwargs)
        

class OfficeHomeReal(OfficeHome):
    def __init__(self, **kwargs):
        super().__init__(subtype='Real World', **kwargs)

