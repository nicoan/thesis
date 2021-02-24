#!/bin/python
import cv2
import numpy as np
import h5py
import os
import cPickle
import sys
from datetime import datetime
from random import shuffle

VERBOSE = True
BATCH_SIZE = 1000
DEFAULT_WIDTH = 64
DEFAULT_HEIGHT = 64
CHANNELS = 4
#BASE_SHAPE = (4, DEFAULT_WIDTH, DEFAULT_HEIGHT) # Corresponds to channels, width, heigth respectively 

# For resnet
OUTPUT_DIM = CHANNELS * DEFAULT_HEIGHT * DEFAULT_WIDTH
BASE_SHAPE = (OUTPUT_DIM, )

def unpickle(file):
    with open(file, 'rb') as fo:
        res = cPickle.load(fo)
    return res

def pickle(data, file):
    with open(file, 'wb') as f:
        res = cPickle.dump(data, f)

def printv(str, *args):
    if VERBOSE:
        print str % args

def format_timedelta(td):
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    hours, minutes, seconds = int(hours), int(minutes), int(seconds)
    if hours < 10:
        hours = '0%s' % int(hours)
    if minutes < 10:
        minutes = '0%s' % minutes
    if seconds < 10:
        seconds = '0%s' % seconds
    return '%s:%s:%s' % (hours, minutes, seconds)

def _change_range(value, old_min=0, old_max=100, new_min=0, new_max=255):
    """
    Changes a number n in (a, b) to a number m in (c, d) mantaining ratio
    """
    old_range = old_max - old_min
    new_range = new_max - new_min
    return (((value - old_min) * new_range) / old_range) + new_min

# So we can apply to numpy arrays
change_range = np.vectorize(_change_range)

def split_image(image, width=64, heigth=64):
    """
    Change the image size and separates it in three channels and returns a the 
    three separated channels
    """
    try:
        img = cv2.imread(image)
        img = cv2.resize(img, (width, heigth), interpolation=cv2.INTER_AREA)
        b, g, r = cv2.split(img)
        return [r, g, b]
    except Exception as e:
        printv('Failed to process image')
        return None

def shape_tuple(size):
    """
    Gets the shape tyuple with size as first argumet
    :param size: Size parameter in shape (how many are in the array)
    :return: (size,) + BASE_SHAPE
    """
    return (size,) + BASE_SHAPE

def save_chunk(dataset, batch, size):
    """
    Append batch of size 'size' to dataset
    """
    dataset_size = len(dataset)
    dataset.resize(shape_tuple(dataset_size + size))
    dataset[dataset_size:] = batch

def get_all_files(directory):
    total_files = []
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            f_name = os.path.join(root, name)
            total_files.append({
                'image': f_name,
                'segment': f_name.replace('.jpg', '.png').replace('UnsupVideo_Frames', 'UnsupVideo_Segments')
            })

    return total_files


def create_dataset(directory, destiny):
    processed = 0
    batch = []
    with h5py.File(destiny, 'w') as hf:
        dataset = hf.create_dataset('dataset',
                                    data=np.empty(shape_tuple(0), dtype='uint8'),
                                    maxshape=shape_tuple(None))

        printv("Loading files...")
        #import pdb; pdb.set_trace()
        files_pickle = './files_array_pickled'
        shuffled_files_pickle = './shuffled_files_array_pickled'
        files = []
        # First we check if we have an array with the files of the dataset saved
        if os.path.exists(shuffled_files_pickle):
            files = unpickle(shuffled_files_pickle)
        else:
            files = get_all_files(directory)
            pickle(files, files_pickle)
            shuffle(files)
            pickle(files, shuffled_files_pickle)
        
        printv("Starting processing images...")
        total_time = datetime.now()
        iteration_time = datetime.now()
        for image_path in files:
            try:
                image = split_image(image_path['image'])
                if CHANNELS == 4:
                    segment_image = split_image(image_path['segment'])[0]
                    segment_image = change_range(segment_image).astype('uint8')
                    image.append(segment_image)
            except:
                printv("Skipped %s due to an error", image_path['image'])
                continue
            if image is not None:
                batch.append(np.array(image).reshape(OUTPUT_DIM))
                #batch.append(np.array(image)) la del reshape es para la resnet
                processed += 1
                # If we already processed BATCH_SIZE images, we save them to the file
                # and reset the array (memory issues)
                if len(batch) == BATCH_SIZE:
                    save_chunk(dataset, batch, BATCH_SIZE)
                    batch = []
                    printv('Processed %s files in %s time' % (processed, format_timedelta(datetime.now() - iteration_time)))
                    iteration_time = datetime.now()
                    sys.stdout.flush()
            #if processed == 5000:
            #    return

        # If some files left in batch unprocessed, we add them too
        save_chunk(dataset, batch, len(batch))
        printv('Finished processing images - time elapsed: %s' % (format_timedelta(datetime.now() - total_time)))

def save_to_file(batch, filename):
    np.save(filename, batch, allow_pickle=False)

if __name__ == '__main__':
    source = '/home/nantinori/tesina/dataset/UnsupVideo_Frames'
    destiny = '/home/nantinori/tesina/dataset/big_dataset_4channel.npy'
    create_dataset(source, destiny)
