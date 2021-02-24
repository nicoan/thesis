import h5py
import numpy as np
import tensorflow as tf
import tflib.inception_score as insc

DATASET_3CHAN = '/home/lilo/Downloads/big_dataset_3channel.npy'
DATASET_4CHAN = '/home/lilo/Downloads/big_dataset_4channel.npy'

# Test score with higher channels
def test_nchan_score(dataset_3chan, dataset_4chan, n):
    for i in xrange(dataset_3chan.shape[0]/n):
        # Prepare 3 chan 
        inp = dataset_3chan[i*n:(i+1)*n]
        inp = inp.reshape(-1, 3, 64, 64).transpose(0,2,3,1)

        #Prepare 4 chan
        inp2 = dataset_4chan[i*n:(i+1)*n]
        inp2 = inp2.reshape(-1, 4, 64, 64)
        # Do the same transpose the generator does when it is marked for scoring
        # take out the 4th channel to evaluate only rgb
        inp2 = tf.Session().run(tf.transpose(tf.transpose(inp2, (1, 0, 2, 3))[:3], (1, 2, 3, 0)))

        # score1 is the original algorithm
        score1 = insc.get_inception_score(list(inp))
        score2 = insc.get_inception_score_lazy2(inp)
        score3 = insc.get_inception_score_lazy2(inp2)
        print( "Original: %s\n3 channel %s\n4 channel %s\nDifference %s %s\n" % (score1[0], score2[0], score3[0], abs(score3[0] - score2[0]), "- Different values!" if score3[0] != score2[0] else ""))


# Test if lazy score is ok
def test_lazy_score(dataset_3chan, n):
    for i in xrange(dataset_3chan.shape[0]/n):
        inp = dataset_3chan[i*n:(i+1)*n]
        inp = inp.reshape(-1, 3, 64, 64).transpose(0,2,3,1)
        score1 = insc.get_inception_score(list(inp))
        score2 = insc.get_inception_score_lazy2(inp)
        print( "Original:%s\nLazy: %s\nDifference: %s %s\n" % (score1[0], score2[0], abs(score1[0] - score2[0]), "- Different values!" if score1[0] != score2[0] else ""))

#python test.py > full_data_inception.txt && systemctl poweroff -i
if __name__ == "__main__":
    with h5py.File(DATASET_3CHAN, 'r') as hf, h5py.File(DATASET_4CHAN, 'r') as hf2:
        inps = hf['dataset']
        inps2 = hf2['dataset']

        print ("Testing if lazy score is ok...")
        test_lazy_score(inps, 1000)

        print "Testing if score with higher channels is ok..."
        test_nchan_score(inps, inps2, 1000)

    print "Calculating inception score for full dataset..."
    full_dataset_inceptionscore = insc.get_inception_score_lazy_full_dataset(DATASET_3CHAN, 'dataset')
    print full_dataset_inceptionscore


