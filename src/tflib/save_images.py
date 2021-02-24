"""
Image grid saver, based on color_grid_vis from github.com/Newmu
"""

import numpy as np
import scipy.misc
from scipy.misc import imsave

# Removes the 4th channel of the image
def remove_alpha(img):
    return map(lambda x: np.delete(x, np.s_[3:], 1), img)

# Cuts the image to focus on the mask it generated (the threshold is 128 by default)
# We take the avergae of the alpha channel for every pixel and we use that as theshold
# (if alpha > alpha_average, we leave the pixel untouched, else we set it to 0)
def cut_mask(img, depth=128):
    h, w = img.shape[:2]
    new_img = np.zeros((h, w, 3))
    for n, x in enumerate(img):
        for m, y in enumerate(x):
            new_img[n, m] = y[:3] if y[3] > depth else np.zeros(3)
    
    return new_img

def save_images(X, save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples/rows

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    # By default there are 3 channels
    channels = 3
    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        channels = X[0].shape[-1]
        img = np.zeros((h*nh, w*nw, channels))
        if channels == 4:
            img_no_alpha = np.zeros((h*nh, w*nw, 3))
            img_no_alpha_mask = np.zeros((h*nh, w*nw, 3))

    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))

    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x
        if channels == 4:
            img_no_alpha[j*h:j*h+h, i*w:i*w+w] = remove_alpha(x)
            img_no_alpha_mask[j*h:j*h+h, i*w:i*w+w] = cut_mask(x)

    imsave(save_path, img)
    if channels == 4:
        imsave(save_path.replace('.png', '_no_alpha.png'), img_no_alpha)
        imsave(save_path.replace('.png', '_mask.png'), img_no_alpha_mask)


def save_images_ORIGINAL(X, save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples/rows

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    #ORIGINAL CODE
    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 4))
        img_no_alpha = np.zeros((h*nh, w*nw, 3))
        img_no_alpha_mask = np.zeros((h*nh, w*nw, 3))

    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))

    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x
        if X.ndim == 4:
            img_no_alpha[j*h:j*h+h, i*w:i*w+w] = remove_alpha(x)
            img_no_alpha_mask[j*h:j*h+h, i*w:i*w+w] = cut_mask(x)

    imsave(save_path, img)
    imsave(save_path.replace('.png', '_no_alpha.png'), img_no_alpha)
    imsave(save_path.replace('.png', '_mask.png'), img_no_alpha_mask)
