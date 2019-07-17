# -*- coding:UTF-8 -*-
######################## DATASET PARAMETERS ###############################

"""
    These values are used for training (when building the dataset)
    patch_size : (x,y) size of the patches to be extracted (usually square)
    step_size : stride of the sliding. If step_size < min(patch_size), then
                there will be an overlap.
"""
#patch_size=(121,121)  #training1128.py
#patch_size=(256,256)
patch_size = (128,128)
step_size = 32

""" ROTATIONS :
    For square patches, valid rotations are 90, 180 and 270.
    e.g. : [] for no rotation, [180] for only 180 rotation, [90, 180]...
"""
ROTATIONS = []
""" FLIPS :
    [False, False] : no symetry
    [True, False] : up/down symetry only
    [False, True] : left/right symetry only
    [True, True] : both up/down and left/right symetries
"""
FLIPS = [False, False]

"""
    BASE_DIR: main dataset folder
    DATASET : dataset name (using for later naming)
    DATASET_DIR : where the current dataset is stored
    FOLDER_SUFFIX : suffix to distinguish this dataset from others (optional)
    BASE_FOLDER : the base folder for the dataset
    BGR : True if we want to reverse the RGB order (Caffe/OpenCV convention)
    label_values : string names for the classes
"""
BASE_DIR = './'
DATASET = 'VOC2012'
FOLDER_SUFFIX = '_fold1'
BASE_FOLDER = BASE_DIR
BGR = True
label_values = ['B-ground','Aero-plane','Bicycle','Bird','Boat','Bottle','Bus','Car','Cat',
               'Chair','Cow','Dining-Table','Dog','Horse','Motorbike','Person',
               'Potted-Plant','Sheep','Sofa','Train','TV','unclassified']
# Color palette
palette = {0: (0,0,0),  # B-ground 
           1: (128, 0, 0),      # Aero-plane 
           2: (0, 128, 0),        # Bicycle 
           3: (128, 128, 0),      # Bird 
           4: (0, 0, 128),    # Boat 
           5: (128, 0, 128),      # Bottle 
           6: (0, 128, 128),    # Bus 
           7: (128, 128, 128),    # Car 
           8: (64, 0, 0),    # Cat 
           9: (192, 0, 0),    # Chair 
           10: (64, 128, 0),    # Cow 
           11: (192, 128, 0),    # Dining-Table 
           12: (64, 0, 128),    # Dog 
           13: (192, 0, 128),    # Horse 
           14: (64, 128, 128),    # Motorbike 
           15: (192, 128, 128),    # Person 
           16: (0, 64, 0),    # Potted-Plant 
           17: (128, 64, 0),    # Sheep 
           18: (0, 192, 0),    # Sofa 
           19: (128, 192, 0),    # Train 
           20: (0, 64, 128),    # TV
           21:(255,255,255)}    #unclassified     


invert_palette = {(0,0,0): 0,  # B-ground 
                  (128, 0, 0): 1,      # Aero-plane 
                  (0, 128, 0):2,         # Bicycle
                  (128, 128, 0): 3,      # Bird 
                  (0, 0, 128): 4,    # Boat 
                  (128, 0, 128): 5,      # Bottle  
		          (0, 128, 128): 6,     # Bus 
                  (128, 128, 128): 7,     # Car 
                  (64, 0, 0): 8,     # Cat 
                  (192, 0, 0): 9,     # Chair 
                  (64, 128, 0): 10,   # Cow 
                  (192, 128, 0): 11,    # Dining-Table 
                  (64, 0, 128): 12,    # Dog 
                  (192, 0, 128): 13,    # Horse 
                  (64, 128, 128): 14,    # Motorbike 
                  (192, 128, 128): 15,    # Person 
                  (0, 64, 0): 16,    # Potted-Plant 
                  (128, 64, 0): 17,   # Sheep 
                  (0, 192, 0): 18,    # Sofa 
                  (128, 192, 0): 19,    # Train 
                  (0, 64, 128): 20,    #TV
                  (255,255,255):21}    # unclassified
                 
                  
NUMBER_OF_CLASSES = len(label_values)

"""
    The folders sequence lists the collections of the dataset, e.g. :
    BaseDirectory/
        MyDataset/
            aux_data/
            data/
            ground_truth/
    Each tuple inside the sequence has :
        (the collection name,
         the subfolder where the collection is stored,
         the filename format)
    For example, if we have : aux_data/aux_1.jpg, aux_data/aux_2.jpg, ...
                              data/1_data.jpg, data/1_data.jpg, ...
                              ground_truth/1_gt.jpg, ground_truth/2_gt.jpg, ...
    The folders variable should look like :
    folders = [
        ('aux_data', BASE_FOLDER + 'aux_data/', 'aux_{}.jpg',
         'data', BASE_FOLDER + 'data/', '{}_data.jpg',
         'ground_truth', BASE_FOLDER + 'ground_truth', '{}_gt.jpg')
    ]
    train_ids and test_ids should detail how to fill the {} in the name format.

    See the examples for the ISPRS Vaihingen and Potsdam for more details.
"""
if DATASET == 'Potsdam':
    folders = [
        ('labels', BASE_FOLDER + 'gts_numpy/', 'top_potsdam_{}_{}_label.png'),
        ('rgb', BASE_FOLDER + '2_Ortho_RGB/', 'top_potsdam_{}_{}_RGB.tif'),
        ('irrg', BASE_FOLDER + 'Y_Ortho_IRRG/', 'top_potsdam_{}_{}_IRRG.tif'),
        ('irgb', BASE_FOLDER + 'X_Ortho_IRGB/', 'top_potsdam_{}_{}_IRGB.tif')
    ]
    train_ids = [
         (3, 12), (6, 8), (4, 11), (3, 10), (7, 9), (4, 10), (6, 10), (7, 7),
         (5, 10), (7, 11), (2, 12), (6, 9), (5, 11), (6, 12), (7, 8), (2, 10),
         (6, 7), (6, 11), (4, 12)]
    test_ids = [(2, 11), (7, 12), (3, 11), (5, 12), (7, 10)]

elif DATASET == 'Vaihingen':
    folders = [
        ('labels', BASE_FOLDER + 'gts_numpy/', 'top_mosaic_09cm_area{}.png'),
        ('irrg', BASE_FOLDER + 'top/', 'top_mosaic_09cm_area{}.tif')
    ]
    train_ids = [(1,), (3,), (5,), (7,), (11,), (13,), (15,),
                 (17,),(21,), (23,), (26,), (28,), (30,)]
    test_ids = [(32,), (34,), (37,)]

elif DATASET == 'zhongwei':
	folders = [
	('labels',BASE_FOLDER + 'matrices/','w_{}'+'.tif'),
	('rgb',BASE_FOLDER + 'zhongwei_origin/new530/','w_{}'+'.png')
]
	train_ids = [
	(1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,),(12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,),(22,),(23,),(24,),(25,),(26,),(27,),(28,),(29,),(30,),(31,),(32,),(33,),(34,),(35,),(36,),(37,),(38,),(39,),(40,),(41,),(42,),(43,),(44,),(45,),(46,),(47,),(48,),(49,),(50,),(51,),(52,),(53,),(54,),(55,),(56,),(57,),(58,),(59,),(60,),(61,),(62,),(63,),(64,),(65,),(66,),(67,),(68,),(69,),(70,),(98,),(99,),(100,),(101,),(102,),(103,),(104,),(105,),(106,),(107,),(108,),(109,),(110,),(111,),(112,),(113,),(114,),(115,),(116,),(117,),(118,),(119,),(120,),(121,),(122,),(123,),(124,),(125,)]
	test_ids = [(71,),(72,),(73,),(74,),(75,),(76,),(77,),(78,),(79,),(80,),(81,),(82,),(83,),(84,),(85,),(86,),(87,),(88,),(89,),(90,),(91,),(92,),(93,),(94,),(95,),(96,),(97,),(126,),(127,),(128,),(129,),(130,),(131,),(132,),(133,)]

elif DATASET == 'VOC2012':
    folders = [
	('labels',BASE_FOLDER + 'matrices/','{}'+'.png'),
	('rgb',BASE_FOLDER + 'ori-voc/','{}'+'.jpg')
]
    train_ids = []
    train_file = open('./train.txt','r')
    train = train_file.readline().strip('\n')
    while train:
        train_ids.append(train)
        train = train_file.readline().strip('\n')

    #print train_ids

    test_ids = []
    test_file = open('./val.txt','r')
    test = test_file.readline().strip('\n')
    while test:
        test_ids.append(test)
        test = test_file.readline().strip('\n')
    
    #print test_ids

# Build the target folder name
DATASET_DIR = BASE_FOLDER + DATASET.lower() + '_{}_{}_{}'.format(
                                    patch_size[0], patch_size[1], step_size)
# Add the suffix is not empty
if FOLDER_SUFFIX:
    DATASET_DIR += FOLDER_SUFFIX

DATASET_DIR += '/'

######################## LMDB PARAMETERS ###############################

""" LMDB to create in a format (source_folder, target_folder) """
#data_lmdbs = [(DATASET_DIR + 'irrg_train/', DATASET_DIR + 'irrg_train_lmdb')]

#test_lmdbs = [(DATASET_DIR + 'irrg_test/', DATASET_DIR + 'irrg_test_lmdb')]

#label_lmdbs = [(DATASET_DIR + 'labels_train/', DATASET_DIR + 'labels_train_lmdb')]

#test_label_lmdbs = [(DATASET_DIR + 'labels_test/', DATASET_DIR + 'labels_test_lmdb')]
data_lmdbs = [(DATASET_DIR + 'rgb_train/',DATASET_DIR + 'rgb_train_lmdb')]

test_lmdbs = [(DATASET_DIR + 'rgb_test/',DATASET_DIR + 'rgb_test_lmdb')]

label_lmdbs = [(DATASET_DIR + 'labels_train/',DATASET_DIR + 'labels_train_lmdb')]

test_label_lmdbs = [(DATASET_DIR + 'labels_test/',DATASET_DIR + 'labels_test_lmdb')]

######################## TESTING PARAMETERS ###############################

"""
    These values are used for testing (when evaluating new data)
    test_patch_size : (x,y) size of the patches to be extracted (usually square)
    test_step_size : stride of the sliding. If step_size < min(patch_size),
                there will be an overlap.
"""
#test_patch_size=(121,121)  #training1128.py
test_patch_size = (256,256)
#test_step_size = 32
test_step_size = 64

######################## CAFFE PARAMETERS ###############################

"""
    CAFFE_ROOT = path to Caffe local installation
    MODEL_FOLDER = where to store the model files
    INIT_MODEL = path to initialization weights (.caffemodel) or None
    CAFFE_MODE = 'gpu' or 'cpu'
    CAFFE_DEVICE = id of the gpu to use (if CAFFE_MODE is 'gpu')
    IGNORE_LABEL = label to ignore when classifying (e.g. clutter)
    TRAIN/TEST_DATA/LABEL_SOURCE = the LMDB containing train/test data and labels
    MEAN_PIXEL = the mean pixel to remove as data normalization (or None)
    BATCH_SIZE = batch size of the network (adjust according to available memory)
"""
CAFFE_ROOT = '/home/mxx/segnet/caffe/'
SOLVER_FILE = '/home/mxx/segnet/models/solver_isprs_vaihingen_irrg.prototxt'
MODEL_FOLDER = '/home/mxx/segnet/models/'
#INIT_MODEL = '/home/liukai/mxx/notebooks/zhongwei/trained_network_weights/potsdam_rgb_128_fold1_iter_80000.caffemodel'
CAFFE_MODE = 'gpu'
CAFFE_DEVICE = 0
IGNORE_LABEL = 21
#TRAIN_DATA_SOURCE = DATASET_DIR + 'irrg_train_lmdb'
TRAIN_DATA_SOURCE = DATASET_DIR + 'rgb_train_lmdb'
TRAIN_LABEL_SOURCE = DATASET_DIR + 'labels_train_lmdb'
#TEST_DATA_SOURCE = DATASET_DIR + 'irrg_test_lmdb'
TEST_DATA_SOURCE = DATASET_DIR + 'rgb_test_lmdb'
TEST_LABEL_SOURCE = DATASET_DIR + 'labels_test_lmdb'
MEAN_PIXEL = [122.675, 116.669, 104.008]
#MEAN_PIXEL = None
BATCH_SIZE=12
