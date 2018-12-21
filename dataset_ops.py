import os
from astropy.table import Table
from astropy.io import fits
import numpy as np
import tensorflow as tf


def load_ground_based_data(path):
    # TODO: add randomly shuffle
    root_path = path + 'GroundBasedTraining/'
    hdfile = os.path.join(root_path, 'catalogs.hdf5')
    if os.path.isfile(hdfile):
        return Table.read(hdfile, path='/ground')
    else:
        cat = Table.read(root_path + 'classifications.csv')
        ims = np.zeros((20000, 4, 101, 101))
        for i, id in enumerate(cat['ID']):
            for j, b in enumerate(['R', 'I', 'G', 'U']):
                ims[i, j] = fits.getdata(
                    root_path + 'Public/Band' +
                    str(j + 1) + '/imageSDSS_' + b + '-' + str(id) + '.fits'
                )
        cat['image'] = ims
        cat.write(hdfile, path='/ground', append=True)
        return cat


def ground_based_input_fn(params):
    cat = load_ground_based_data(params['gound_based_path'])
    n_train = params['n_data']
    train_x, train_y = [], []
    train_x.append(cat['image'][0:n_train])
    train_y.append(cat['is_lens'][0:n_train])
    train_x = np.vstack(train_x)
    train_y = np.hstack(train_y)
    return train_x, train_y


def load_dataset(db_path, normalize=True, one_hot=False):
    data = Table.read(db_path, path='/test')

    train_x, train_y = [], []
    train_x.append(data['image'][0:_UNIT_NUM_ * _TIMES_])
    train_y.append(data['is_lens'][0:_UNIT_NUM_ * _TIMES_])
    train_x = np.vstack(train_x)
    train_y = np.hstack(train_y)
    #train_x = train_x.astype('float32')
    #train_y = train_y.astype('float32')
    train_x = train_x.reshape((_UNIT_NUM_ * _TIMES_, 1, 48, 48))

    test_x, test_y = [], []
    test_x.append(data['image'][-_UNIT_NUM_:])
    test_y.append(data['is_lens'][-_UNIT_NUM_:])
    test_x = np.vstack(test_x)
    test_y = np.hstack(test_y)
    test_x = test_x.reshape(_UNIT_NUM_, 1, 48, 48)
    return train_x, train_y, test_x, test_y


def preprocess_dataset(savedir, train_x, train_y, test_x, test_y):
    indices = np.arange(len(train_x))
    np.random.shuffle(indices)
    train_x = train_x[indices]
    train_y = train_y[indices]
    train_mask = np.zeros(len(train_y), dtype=np.float32)
    mask_count = int(_UNIT_NUM_ * _TIMES_ * _MASK_RATE_)
    count = [0, 0]
    for i in range(len(train_y)):
        if sum(count) == mask_count:
            break
        label = int(train_y[i])
        if count[label] < (mask_count // 2):
            train_mask[i] = 1.0
            count[label] += 1

    for i in range(len(train_y)):
        if not train_mask[i] > 0:
            train_y[i] = -1.0  # unlabeled

    return train_x, train_y, train_mask, test_x, test_y
