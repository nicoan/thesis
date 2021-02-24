from datetime import datetime

import sys
from pycocotools.coco import COCO
from random import shuffle
import h5py
import numpy as np
import skimage.io as io
import cv2
import os

dataDir = '.'
dataType = 'val2017'
annFile = '{}/instances_{}.json'.format(dataDir, dataType)
imagesDir = '/home/nico/Downloads/val2017'
samplesDir = './segments'
originalImageDir = '{}/original'.format(samplesDir)
segmentedImageDir = '{}/segmented'.format(samplesDir)
WIDTH = 64
HEIGHT = 64
BATCH_SIZE = 1000
CHANNELS = 4
OUTPUT_DIM = CHANNELS * HEIGHT * WIDTH



# cats = coco.loadCats(coco.getCatIds())
# nms=[cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))
#
# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))

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


def shape_tuple(size):
    """
    Gets the shape tyuple with size as first argumet
    :param size: Size parameter in shape (how many are in the array)
    :return: (size,) + BASE_SHAPE
    """
    return (size,) + (OUTPUT_DIM, )#(CHANNELS, WIDTH, HEIGHT)


def save_chunk(dataset, batch, size):
    """
    Append batch of size 'size' to dataset
    """
    dataset_size = len(dataset)
    dataset.resize(shape_tuple(dataset_size + size))
    dataset[dataset_size:] = batch


def get_all_files(directory):
    """
    Gets all files of a directory with the path prepended
    """
    for root, dirs, files in os.walk(directory, topdown=False):
        return map(lambda x: os.path.join(root, x), files)


def get_polygon(points):
    """
    Gets a polygon from a group of points
    """
    polygon = np.array(points)
    return polygon.reshape((len(points) / 2, 2)).astype('int32')


def get_segmentation_percent(segmentation):
    seg = segmentation.reshape((-1,))
    total_pixels = seg.shape[0]
    segment_pixels = len(filter(lambda x: x != 0, segmentation.reshape((-1,))))
    return float(segment_pixels) / float(total_pixels - segment_pixels)


def add_segmentation(original_image, segmentations, theshold=0.15):
    """
    Given a group of segmentations, add an alpha channel to the image
    with them in it. It only adds it if segmentations are al least @theshold
    percent of image
    """
    original_shape = original_image.shape
    segment_channel = np.zeros(original_shape[:2] + (1,), np.uint8)
    try:
        cv2.fillPoly(segment_channel, map(lambda x: get_polygon(x), segmentations), 255)
    except:
        return None
    if get_segmentation_percent(segment_channel) < theshold:
        return None
    return np.concatenate((original_image, segment_channel), axis=2)


def extract_dataset():
    """
    Extracts the images and their segmentations and saves them in separated 
    images resized
    """
    if not os.path.exists(samplesDir):
        os.makedirs(samplesDir)
    if not os.path.exists(originalImageDir):
        os.makedirs(originalImageDir)
    if not os.path.exists(segmentedImageDir):
        os.makedirs(segmentedImageDir)

    coco = COCO(annFile)
    # get all images containing given categories, select one at random
    cat_ids = coco.getCatIds()
    img_ids = coco.getImgIds()
    imgs = coco.loadImgs(ids=img_ids)

    for img in imgs:
        try:
            #   I = io.imread(img['coco_url'])
            I = io.imread('%s/%s' % (imagesDir, img['file_name']))
            # load and display instance annotations
            ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(ann_ids)

            added_segmentation = False
            for i, ann in enumerate(anns):
                segmented_image = add_segmentation(I, ann['segmentation'])
                # Resize to the desired size
                if segmented_image is not None:
                    added_segmentation = True
                    segmented_image = cv2.resize(segmented_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                    io.imsave('%s/%s_%s.png' % (segmentedImageDir, img['id'], i), segmented_image)

            if added_segmentation:
                original_resized = cv2.resize(I, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                io.imsave('%s/%s.jpg' % (originalImageDir, img['id']), original_resized)
        except Exception as e:
            print "There was an error processing some image\n%s" % e
            continue


def make_dataset(directory, name):
    """
    Generates the h5py containing all the images
    """
    files = get_all_files(directory)
    shuffle(files)

    with h5py.File(name, 'w') as hf:
        dataset = hf.create_dataset('dataset', data=np.empty(shape_tuple(0), dtype='uint8'), maxshape=shape_tuple(None))

        processed = 0
        batch = []
        print "Starting processing images..."
        total_time = datetime.now()
        iteration_time = datetime.now()
        for image_path in files:
            try:
                #import pdb;pdb.set_trace()
                image = io.imread(image_path).transpose(2, 0, 1)
            except:
                print "Skipped %s due to an error" % image_path['image']
                continue

            batch.append(image.reshape(OUTPUT_DIM))
            processed += 1
            # If we already processed BATCH_SIZE images, we save them to the file
            # and reset the array (memory issues)
            if len(batch) == BATCH_SIZE:
                save_chunk(dataset, batch, BATCH_SIZE)
                batch = []
                print 'Processed %s files in %s time' % (processed, format_timedelta(datetime.now() - iteration_time))
                iteration_time = datetime.now()
                sys.stdout.flush()

            #if processed == 5000:
            #    return

        # If some files left in batch unprocessed, we add them too
        save_chunk(dataset, batch, len(batch))
        print 'Finished processing images - time elapsed: %s' % (format_timedelta(datetime.now() - total_time))


if __name__ == "__main__":
    # extract_dataset()
    make_dataset('/home/cuda/Shared/segments/segmented', 'coco_4channel.h5p')
