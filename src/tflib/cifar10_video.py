import numpy as np
import scipy.misc
import time
import h5py

def cifar_generator(path, batch_size):
    hf = h5py.File(path, 'r')
    images = hf['dataset']
    total_batches = (len(images) / batch_size) - 1
    epoch_count = [0] 
    def get_epoch():
        epoch_count[0] = 0 if total_batches == epoch_count[0] else epoch_count[0] + 1
        print '%s/%s' % (epoch_count[0], total_batches)
        yield (images[epoch_count[0] * batch_size : (epoch_count[0] + 1) * batch_size], [0] * batch_size)

    return get_epoch

def fake_labels(batch_size):
    def get_labels(): 
        yield ([0] * batch_size, [0] * batch_size)

    return get_labels

def load(batch_size, data_dir='/home/lilo/Downloads/4_channel_dataset.npy'):
    return (cifar_generator(data_dir, batch_size), cifar_generator(data_dir, batch_size))

if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()