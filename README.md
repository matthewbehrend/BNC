# Batch Normalization Classifer

A Batch Normalization Classifier for Domain Adaptation
Matthew R. Behrend and Sean M. Robinson

## Datasets
Manually download the OfficeHome dataset from https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view
Extract the zip file and move contents to ./data/officehome

Initial feature extraction may take some time. The result will be stored in a numpy archive in ./data_cache for fast loading


# Install required packages
developed with python 3.7.9
```bash
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
```

## Usage
To reproduce data from paper
```bash
python run_experiments.py
```

## Contents
/adaptation_methods/adapt_BNC.py contains the model and training methods

class BNC implements the batch normalized classifier and training loop
class BNC_Cotrained uses cotraining on source and target domains for comparison to the source-free adaptation in class BNC

class BNCInspect explores feature distributions in an ablation study of the batchnorm layer


## Reference
Please cite our paper if you use this code
```
M.R. Behrend and S.M. Robinson, "A Batch Normalization Classifier for Domain Adaptation", arXiv e-prints, p. arXiv:2103.11642, 2021.
https://arxiv.org/abs/2103.11642
```

## Contact
Matthew Behrend
behrend04@gmail.com

Sean Robinson
sean@psl.com

## License
MIT License

## Citations
Dataset loading uses portions of code from the following sources: Tzeng2017, Ringwald 2020
Eric Tzeng, Judy Hoffman, Kate Saenko, and Trevor Darrell. Adversarial discriminative domain adaptation. In Proceedings IEEE Conference on Computer Vision and Pattern Recognition. pages 7167-7176, 2017.
Tobias Ringwald and Rainer Stiefelhagen. Unsupervised Domain Adaptation by Uncertain Feature Alignment. arXiv preprint arXiv:2009.06483, 2020.










