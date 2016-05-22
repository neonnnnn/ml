import numpy as np


def tofloatarray(data):
    data = np.asarray(data, dtype=np.float32)
    return data


def tointarray(data):
    data = np.asarray(data, dtype=np.int32)
    return data


def separate(dict):
    x = dict["data"]
    y = dict["labels"]
    return x, y


def load_data():
    def unpickle(f):
        import cPickle
        fo = open(f, 'rb')
        d = cPickle.load(fo)
        fo.close()
        return d

    datadir = "dataset/cifar-10/"
    d1 = unpickle(datadir + "data_batch_1")
    d2 = unpickle(datadir + "data_batch_2")
    d3 = unpickle(datadir + "data_batch_3")
    d4 = unpickle(datadir + "data_batch_4")
    d5 = unpickle(datadir + "data_batch_5")
    d6 = unpickle(datadir + "test_batch")

    data1, labels1 = separate(d1)
    data2, labels2 = separate(d2)
    data3, labels3 = separate(d3)
    data4, labels4 = separate(d4)
    data5, labels5 = separate(d5)

    train_x = np.vstack((data1, data2))
    train_y = labels1 + labels2
    train_x = np.vstack((train_x, data3))
    train_y = train_y + labels3
    train_x = np.vstack((train_x, data4))
    train_y = train_y + labels4
    train_x = np.vstack((train_x, data5))
    train_y = train_y + labels5
    test_x, test_y = separate(d6)

    train_set_x = tofloatarray(train_x)
    train_set_y = tointarray(train_y)
    test_set_x = tofloatarray(test_x)
    test_set_y = tointarray(test_y)

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval
