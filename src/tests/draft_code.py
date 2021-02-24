
    ## Inception score codes

    # For comparison
    def get_inception_score(n):
        all_samples = []

        # Removemos el archivo que contiene las imagenes para calcular el inception score
        if (os.path.exists(LAZY_INCEPTION_SCORE_FILE)):
            os.remove(LAZY_INCEPTION_SCORE_FILE)

        if (os.path.exists(LAZY_INCEPTION_SCORE_FILE)):
            os.remove(LAZY_INCEPTION_SCORE_FILE + "2")

        # Lo creamos de nuevo con las nuevas imagenes
        with h5py.File(LAZY_INCEPTION_SCORE_FILE, 'w') as hf, h5py.File(LAZY_INCEPTION_SCORE_FILE + "2", 'w') as hf2:
            inception_dataset = hf.create_dataset('inception_dataset', data=np.empty((0, INPUT_WIDTH, INPUT_HEIGHT, 3), dtype='int32'), maxshape=(None, INPUT_WIDTH, INPUT_HEIGHT, 3))
            inception_dataset2 = hf2.create_dataset('inception_dataset', data=np.empty((0, INPUT_WIDTH, INPUT_HEIGHT, 3), dtype='int32'), maxshape=(None, INPUT_WIDTH, INPUT_HEIGHT, 3))
            for i in xrange(n/100):
                (batch_sample1, batch_sample2) = session.run(samples_100)

                # batch 1
                batch_sample1 = np.floor(((batch_sample1+1.)*(255.99/2)))
                save_chunk(inception_dataset, batch_sample1, len(batch_sample1))

                # batch 2
                batch_sample2 = np.floor(((batch_sample2+1.)*(255.99/2)))
                batch_sample2 = batch_sample2.reshape((-1, 3, INPUT_WIDTH, INPUT_HEIGHT)).transpose(0,2,3,1)
                save_chunk(inception_dataset2, batch_sample2, len(batch_sample2))

        result1 = lib.inception_score.get_inception_score_lazy(LAZY_INCEPTION_SCORE_FILE)
        result2 = lib.inception_score.get_inception_score_lazy(LAZY_INCEPTION_SCORE_FILE + "2")
        print "Result 1: %s\nResult 2:%s\n" % (result1, result2)
        return result1

    # Original

    def get_inception_score(n):
       all_samples = []
       import pdb; pdb.set_trace()
       for i in xrange(n/100):
           all_samples.append(session.run(samples_100))
       all_samples = np.concatenate(all_samples, axis=0)
       all_samples = ((all_samples+1.)*(255.99/2)).astype('int32')
       all_samples = all_samples.reshape((-1, 3, INPUT_WIDTH, INPUT_HEIGHT)).transpose(0,2,3,1)
       return lib.inception_score.get_inception_score(list(all_samples))



# Used as comparison for the real function
def get_inception_score_lazy2(inps, splits=10):
  #assert(type(images) == list)
  #assert(type(images[0]) == np.ndarray)
  #assert(len(images[0].shape) == 3)
  #assert(np.max(images[0]) > 10)
  #assert(np.min(images[0]) >= 0.0)
  
  #inps = []
  #for img in images:
  #  img = img.astype(np.float32)
  #  inps.append(np.expand_dims(img, 0))

  bs = 100
  with tf.Session() as sess:
    preds = []
    n_batches = int(math.ceil(float(inps.shape[0]) / float(bs)))
    for i in range(n_batches):
        # sys.stdout.write(".")
        # sys.stdout.flush()
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]

        pred = sess.run(softmax, {'ExpandDims:0': inp})
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)
