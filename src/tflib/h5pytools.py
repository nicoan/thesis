def save_chunk(dataset, batch, shape=(64, 64, 3)):
    """
    Append batch of size 'size' to dataset
    """
    dataset_size = len(dataset)
    dataset.resize((dataset_size + len(batch),) + shape)
    dataset[dataset_size:] = batch