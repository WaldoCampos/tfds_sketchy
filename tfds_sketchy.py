import tensorflow_datasets as tfds
import os
import skimage.io as io
import numpy as np
import json

_CATEGORIES_FILE = 'sketchy_classes.json'
_TRAIN_FILE = './train.txt'
_VALID_KNOWN_FILE = './valid_known.txt'
_VALID_UNKNOWN_FILE = './valid_unknown.txt'

class tfds_sketchy(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')

    def get_categories(self):
        with open(_CATEGORIES_FILE) as json_file:
            data = json.load(json_file)
        categories = data.keys()
        return categories

    def _info(self):
        categories = self.get_categories()
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'sketch': tfds.features.Image(shape=(256, 256, 3)),
                'photo': tfds.features.Image(shape=(256, 256, 3)),
                'label': tfds.features.ClassLabel(names=categories),
            }),
            supervised_keys=('sketch', 'photo', 'label'),
            )
    
    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            'train': self._generate_examples(_TRAIN_FILE),
            'validation_known': self._generate_examples(_VALID_KNOWN_FILE),
            'validation_unknown': self._generate_examples(_VALID_UNKNOWN_FILE),
            }
    
    def _generate_examples(self, fname):
        with open(fname) as flist :
            for i , f in enumerate(flist):       
                if (i + 1)  % 10000 == 0 :
                    print('{} {}'.format(fname, i+1))                                      
                data = f.strip().split('\t')
                sketch_path = data[0].strip()
                sketch = io.imread(sketch_path)
                photo_path = data[1].strip()
                photo = io.imread(photo_path)
                label = int(data[2].strip())

                yield i, {
                    'sketch': sketch,
                    'photo': photo,
                    'label': label,
                }
