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
