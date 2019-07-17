
# coding: utf-8

# In[1]:
""" inference.py
    This scripts tests a SegNet model using a predefined Caffe solver file.
    Args: the weights (.caffemodel file) to use and the ids of the tiles to
    process
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from skimage import img_as_float, io
from sklearn.metrics import confusion_matrix
import itertools
import argparse
import os
from tqdm import tqdm


from config import CAFFE_ROOT, MODEL_FOLDER, CAFFE_MODE, CAFFE_DEVICE,\
                   TRAIN_DATA_SOURCE, TRAIN_LABEL_SOURCE,\
                   TEST_DATA_SOURCE, TEST_LABEL_SOURCE, MEAN_PIXEL,\
                   BATCH_SIZE, BGR, label_values, BASE_FOLDER,\
                   test_patch_size, test_step_size

#print CAFFE_ROOT
from training0424 import segnet_network

import caffe
from caffe import layers as L, params as P


sys.path.insert(0, CAFFE_ROOT + 'python/')

plt.rcParams['figure.figsize'] = (15,15)

def label_to_pixel(label):
    """ Converts the numeric label from the ISPRS dataset into its RGB encoding

    Args:
        label (int): the label value (numeric)

    Returns:
        numpy array: the RGB value
    """
    codes = [[0,0,0],  # B-ground 
            [128, 0, 0],      # Aero-plane 
            [0, 128, 0],        # Bicycle 
            [128, 128, 0],      # Bird 
            [0, 0, 128],    # Boat 
            [128, 0, 128],      # Bottle 
            [0, 128, 128],    # Bus 
            [128, 128, 128],    # Car 
            [64, 0, 0],    # Cat 
            [192, 0, 0],    # Chair 
            [64, 128, 0],    # Cow 
            [192, 128, 0],    # Dining-Table 
            [64, 0, 128],    # Dog 
            [192, 0, 128],    # Horse 
            [64, 128, 128],    # Motorbike 
            [192, 128, 128],    # Person 
            [0, 64, 0],    # Potted-Plant 
            [128, 64, 0],    # Sheep 
            [0, 192, 0],    # Sofa 
            [128, 192, 0],    # Train 
            [0, 64, 128],     # TV
            [255,255,255]]    #unclassified
        
    return np.asarray(codes[int(label)])

def prediction_to_image(prediction, reshape=False):
    """ Converts a prediction map to the RGB image

    Args:
        prediction (array): the input map to convert
        reshape (bool, optional): True if reshape the input from Caffe format
                                  to numpy standard 2D array

    Returns:
        array: RGB-encoded array
    """
    if reshape:
        prediction = np.swapaxes(np.swapaxes(prediction, 0, 2), 0, 1)
    image = np.zeros(prediction.shape[:2] + (3,), dtype='uint8')
    for x in xrange(prediction.shape[0]):
        for y in xrange(prediction.shape[1]):
            image[x,y] = label_to_pixel(prediction[x,y])
    return image


# In[6]:

# Simple sliding window function
def sliding_window(top, step=10, window_size=(20,20)):
    """Extract patches according to a sliding window.

    Args:
        image (numpy array): The image to be processed.
        stride (int, optional): The sliding window stride (defaults to 10px).
        window_size(int, int, optional): The patch size (defaults to (20,20)).

    Returns:
        list: list of patches with window_size dimensions
    """
    # slide a window across the image
    for x in xrange(0, top.shape[0], step):#xrange用法和range相同，但是xrange生成的是生成器，range生成的数组，xrange省内存
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in xrange(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]
 
def count_sliding_window(top, step=10, window_size=(20,20)):
    """Count the number of patches in a sliding window.

    Args:
        image (numpy array): The image to be processed.
        stride (int, optional): The sliding window stride (defaults to 10px).
        window_size(int, int, optional): The patch size (defaults to (20,20)).

    Returns:
        int: patches count in the sliding window
    """
    c = 0
    # slide a window across the image
    for x in xrange(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in xrange(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c

def process_patches(images, net, transformer):
    """ Process a patch through the neural network and extract the predictions

    Args:
        images (array list): list of images to process (length = batch_size)
        net (obj): the Caffe Net
        transformer (obj): the Caffe Transformer for preprocessing
    """
    # caffe.io.load_image converts to [0,1], so our transformer sets it back to [0,255]
    # but the skimage lib already works with [0, 255] so we convert it to float with img_as_float
    data = np.zeros(net.blobs['data'].data.shape)
    for i in range(len(images)):
        data[i] = transformer.preprocess('data', img_as_float(images[i]))
    net.forward(data=data)
    output = net.blobs['conv1_1_D'].data[:len(images)]
    output = np.swapaxes(np.swapaxes(output, 1, 3), 1, 2)
    return output

def grouper(n, iterable):
    """ Groups elements in a iterable by n elements

    Args:
        n (int): number of elements to regroup
        iterable (iter): an iterable

    Returns:
        tuple: next n elements from the iterable
    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))#islice:切片，截取迭代器的一部分，步长为n
        if not chunk:
            return
        yield chunk


# In[7]:

def predict(image, net, transformer, step=32, patch_size=(128,128)):
    """Generates prediction from a tile by sliding window and neural network

    Args:
        image (array): the tile to be processed
        step (int, optional): the stride of the sliding window
        patch_size (int tuple, optional): the dimensions of the sliding window

    Returns:
        votes (array): predictions for the tile
    """
    votes = np.zeros(image.shape[:2] + (22,))#4表示类别数目
    for coords in tqdm(grouper(BATCH_SIZE, sliding_window(image, step, patch_size)), total=count_sliding_window(image, step, patch_size)/BATCH_SIZE + 1):
        image_patches = []

        for x,y,w,h in coords:
            image_patches.append(image[x:x+w, y:y+h])

        predictions = process_patches(image_patches, net, transformer)
        for (x,y,w,h), prediction in zip(coords, predictions):
            for i in xrange(x, x+w-1):
                for j in xrange(y, y+h-1):
                    votes[i,j] += prediction[i-x, j-y]
    return votes

def process_votes(prediction):
    """ Returns RGB encoded prediction map

    Args:
        votes (array): full prediction from the predict function

    Returns:
        array: RGB encoded prediction
    """
    rgb = np.zeros(prediction.shape[:2] + (3,), dtype='uint8')
    for x in xrange(prediction.shape[0]):
        for y in xrange(prediction.shape[1]):
            rgb[x,y] = np.asarray(label_to_pixel(np.argmax(prediction[x,y])))
    return rgb

def pixel_to_label(pixel):
    """ Convert RGB pixel value of a label to its numeric id

    Args:
        pixel (array): RGB tuple of the pixel value
    Returns:
        int: label id
    """
    label = None
    # Code for RGB values to label :
    r, g, b = pixel
    if r == 0 and g == 0 and b == 0:
        label = 0 # B-ground 
    elif r == 128 and g == 0 and b == 0:
        label = 1  # Aero-plane
    elif r == 0 and g == 128 and b == 0:
        label = 2 # Bicycle
    elif r == 128 and g == 128 and b == 0:
        label = 3 # # Bird
    elif r == 0 and g == 0 and b == 128:
        label = 4 # Boat
    elif r == 128 and g == 0 and b == 128:
        label = 5 # Bottle
    elif r == 0 and g == 128 and b == 128:
        label = 6 # Bus
    elif r == 128 and g == 128 and b == 128:
        label = 7 # Car
    elif r == 64 and g == 0 and b == 0:
        label = 8 # Cat
    elif r == 192 and g == 0 and b == 0:
        label = 9 # Chair
    elif r == 64 and g == 128 and b == 0:
        label = 10 # Cow
    elif r == 192 and g == 128 and b == 0:
        label = 11 # Dining-Table
    elif r == 64 and g == 0 and b == 128:
        label = 12 # Dog  
    elif r == 192 and g == 0 and b == 128:
        label = 13 # Horse
    elif r == 64 and g == 128 and b == 128:
        label = 14 # Motorbike 
    elif r == 192 and g == 128 and b == 128:
        label = 15 # Person 
    elif r == 0 and g == 64 and b == 0:
        label = 16 # Potted-Plant 
    elif r == 128 and g == 64 and b == 0:
        label = 17 # Sheep 
    elif r == 0 and g == 192 and b == 0:
        label = 18 # Sofa 
    elif r == 128 and g == 192 and b == 0:
        label = 19 # Train  
    elif r == 0 and g == 64 and b == 128:
        label = 20 # TV  
    elif r == 255 and g == 255 and b == 255:
        label = 21 # unclassified    
    return label

    
def flatten_predictions(prediction, gt):
    """ Converts the RGB-encoded predictions and ground truth into the flat
        predictions vectors used to compute the confusion matrix

    Args:
        prediction (array): the RGB-encoded prediction
        gt (array): the RGB-encoded ground truth

    Returns:
        array, array: the flattened predictions, the flattened ground truthes

    """
    gt_labels = np.zeros(gt.shape[:2])
    prediction_labels = np.zeros(prediction.shape[:2])
    for l_id, label in enumerate(label_values):
        r,g,b = label_to_pixel(l_id)
        mask = np.logical_and(gt[:,:,0] == r, gt[:,:,1] == g)
        mask = np.logical_and(mask, gt[:,:,2] == b)
        gt_labels[mask] = l_id
        
        mask = np.logical_and(prediction[:,:,0] == r, prediction[:,:,1] == g)
        mask = np.logical_and(mask, prediction[:,:,2] == b)
        prediction_labels[mask] = l_id
       

    return prediction_labels.flatten(), gt_labels.flatten()#flatten()返回折叠为一维的数组

def metrics(predictions, gts):
    """ Compute the metrics from the RGB-encoded predictions and ground truthes

    Args:
        predictions (array list): list of RGB-encoded predictions (2D maps)
        gts (array list): list of RGB-encoded ground truthes (2D maps, same dims)
    """
    labels = [flatten_predictions(prediction, gt) for prediction, gt in zip(predictions, gts)]
    prediction_labels = np.concatenate([label[0] for label in labels])
    gt_labels = np.concatenate([label[1] for label in labels])
    #print prediction_labels
    #print prediction
    #print gt_labels
    #confusion_matrix：混淆矩阵，用于给出分类模型预测结果的情形分析表
    cm = confusion_matrix(
            gt_labels,
            prediction_labels,
            range(len(label_values)))

    print "Confusion matrix :"
    #print cm
    print "---"
    # Compute global accuracy
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    total = sum(sum(cm))#先列求和，再行求和
    print "{} pixels processed".format(total)
    print "Total accuracy : {}%".format(accuracy * 100 / float(total))
    print "---"
    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in xrange(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print "F1Score :"
    for l_id, score in enumerate(F1Score):
        print "{}: {}".format(label_values[l_id], score)
    print "---"

    #Compute MIOU
    IOU = np.zeros(len(label_values))
    MIOU = 0
    for i in xrange(len(label_values)):
        try:
            IOU[i] = 1. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]) - cm[i,i])
        except:
            pass
    print "IOU:"
    for l_id, score in enumerate(IOU):
        print "{}: {}".format(label_values[l_id], score)
    

    print "MIOU:"
    MIOU = np.sum(IOU) / 21
    print MIOU
    print "---"
    
    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
    kappa = (pa - pe) / (1 - pe);
    print "Kappa: " + str(kappa)


# In[ ]:
def main(weights, infer_ids, save_dir):
    # Caffe configuration : GPU and use device 0
    n = caffe.NetSpec()
    if CAFFE_MODE == 'gpu':
        caffe.set_mode_gpu()
        caffe.set_device(CAFFE_DEVICE)
    else:
        caffe.set_mode_cpu()

    net_arch = segnet_network(TEST_DATA_SOURCE, TEST_LABEL_SOURCE, mode='deploy')
    # Write the train prototxt in a file
    f = open(MODEL_FOLDER + 'test_segnet.prototxt', 'w')
    f.write(str(net_arch.to_proto()))
    f.close()
    print "Caffe definition prototxt written in {}.".format(MODEL_FOLDER + 'test_segnet.prototxt')

    net = caffe.Net(MODEL_FOLDER + 'test_segnet.prototxt',
                    weights,
                    caffe.TEST)

# In[4]:

    """ Defines the Caffe transformer that will be able to preprocress the data
        before feeding it to the neural network
    """
    # Initialize the transformer
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # Normalize the data by substracting the mean pixel
    transformer.set_mean('data', np.asarray(MEAN_PIXEL))
    # Reshape the data from numpy to Caffe format (channels first : WxHxC -> CxWxH)
    transformer.set_transpose('data', (2,0,1))
    # Data is expected to be in [0, 255] (int 8bit encoding)
    transformer.set_raw_scale('data', 255.0)
    # Transform from RGB to BGR
    if BGR:
        transformer.set_channel_swap('data', (2,1,0))

    print infer_ids
    pictures=open(infer_ids[0],'r')
    pic_lines=[]
    pic_line=pictures.readline().strip('\n')
    while pic_line:
        pic_lines.append(pic_line)
        pic_line=pictures.readline().strip('\n')


    imgs = [io.imread(BASE_FOLDER + 'ori-voc/' + '{}.jpg'.format(l)) for l in pic_lines]
    print "Processing {} images...".format(len(imgs))
    predictions = [process_votes(predict(img, net, transformer, step=test_step_size, patch_size=test_patch_size)) for img in imgs]
    
    

    results = []
    for pred, id_ in zip(predictions, pic_lines):
        filename = save_dir + 'voc-result/segnet_voc_{}x{}_{}_area{}.png'.format(\
                test_patch_size[0], test_patch_size[1], test_step_size, id_)
        io.imsave(filename, pred)
       # print "Results for tile {} saved in {}".format(id_, filename)
        results.append((pred, filename))

    gts = [io.imread(BASE_FOLDER + 'gts-voc/' + '{}.png'.format(l)) for l in pic_lines]

    print "Computing metrics..."
    metrics(predictions, gts)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SegNet Inference Script')
    parser.add_argument('infer_ids', type=str, metavar='tiles', nargs='+',
                        help='id of tiles to be processed')
    parser.add_argument('--weights', type=str, required=True,
                       help='path to the caffemodel file with the trained weights')
    parser.add_argument('--dir', type=str,
                       help='Folder where to save the results')
    args = parser.parse_args()
    weights = args.weights
    infer_ids = args.infer_ids
    save_dir = args.dir
    if save_dir is None:
        save_dir = '/home/mxx/segnet/notebooks/VOC2012/'
    main(weights, infer_ids, save_dir)
