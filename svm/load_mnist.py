import cPickle
import gzip
import numpy as np


def load_data(filepath='../../dataset/mnist.pkl.gz'):
    print 'loading data...'

    # Load the dataset
    f = gzip.open(filepath, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        data_x = np.asarray(data_x, dtype=np.float32)
        data_y = np.asarray(data_y, dtype=np.int32)

        return data_x, data_y

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    print 'loading complete'
    return rval