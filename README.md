# Fine grained classification using BCNN
This is a simple pytorch re-implementation of ICCV 2015 [Bilinear CNN Models for Fine-grained Visual Recognition](http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf).

### Introduction:
The features are summarized below:
+ Use VGG16 as base Network.

### Usage:

+ Define class names in data_augment.py and test.py
+ Follow the directory and file structure as defined in data_augment.py

+ Put the data in Data/original and test data in Data/test

+ Data augmentation
``` 
 python2 data_aug.py --data path/to/original/data --save path/to/dir/to/save/images

```
+ Divide into train-test sets
```
python2 divide.py --aug_data path/to/augmented/data --save dir/to/save/train/test/files --ratio train/percentage
```
+ Train the model
```
 CUDA_VISIBLE_DEVICES=1 python2 train.py --base_lr lr --batch_size bs --epochs epoch --weight_decay decay --aug_data path/to/dir/containing/augdata --text_path /path/to/train/text/file --save_model /path/to/model

```
+ Test
```
 CUDA_VISIBLE_DEVICES=1 python2 test.py --model_path path/to/saved/model --test_path /path/to/dir/containing/testdata --output_path path/to/dir/to/save/output
 ```
 

