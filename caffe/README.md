<<<<<<< HEAD
# Improved SegNet-caffe

# platform
ubuntu16.4+cuda 9.0+cudnn7.1+opencv3.1+caffe+python2.7

# requirments
lmdb==0.89
matplotlib==1.5.2
numpy==1.11.1
scipy==0.18.0
tqdm==4.8.1
protobuf==3.0.0
Cython==0.24.1
Pillow==3.3.1
scikit_image==0.12.3
scikit_learn==0.17.1
pydot==1.2.2

# How to start

  1. First, we need to edit the `config.py` file. `BASE_DIR`, `DATASET` and `DATASET_DIR` are used to point to the folder where the dataset is stored and to specify a unique name for the dataset, e.g. "VOC2012". `label_values` and `palette` define the classes that will be used and the associated colors in RGB format. `folders`, `train_ids` and `test_ids` define the folder arrangement of the dataset and the train/test split using unique numerical ids associated to the tiles.
  2. We need to transform the ground truth RGB-encoded images to 2D matrices. We can use the `convert_gt.py` script to do so, e.g. : `python convert_gt.py gts_voc/*.tif --from-color --out matrices/`. This will populate a new `matrices/` folder containing the matrices. Please note that the `folders` value for the labels should point to this folder (`matrices/`).
  3. Extract small patches from the tiles to create the train and test sets : `python extract_images.py`
  4. Populate LMDB using the extracted images : `python create_lmdb.py`
  5. Train the network for 40 000 iterations, starting with VGG-16 weights and save the weights into the `trained_network_weights` folder : `python training.py --niter 40000 --update 1000 --init vgg16weights.caffemodel --snapshot trained_network_weights/`
  6. Test the trained network on some tiles : `python inference.py pictures.txt --weights trained_network_weights/net_iter_40000.caffemodel`."pictures.txt" stores ids of test pictures.
=======
# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
>>>>>>> caffe
