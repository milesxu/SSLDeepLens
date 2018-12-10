import math
import numpy as np
import tensorflow as tf
from absl import app as absl_app
from absl import flags
from astropy.table import Table
import resnet_model_ssl
from official.resnet import resnet_run_loop
from official.utils.logs import logger


_DB_PATH_1_ = '/home/milesx/Downloads/catalogs_test_ones_and_twos_40000.hdf5'
_UNIT_NUM_ = 1000
_TIMES_ = 3
_SAVE_DIR_ = '.'
_MASK_RATE_ = 0.5

tempens_params = {
    'type': 'tempens',
    'masks': [],
    'ensemble_pred': [],
    'targets': [],
    'lr': 0.003,
    'adam_beta1': 0.9,
    'rd_beta_target': 0.5,
    'scale_unsup_wght_max': 100.0,
    'coeff_embed': 0.2,
    'num_epochs': 300,
    'rampup': 80,
    'rampdown': 50,
}


def params_update(params):

    def rampup(epoch):
        p = tf.cast(epoch, tf.float32) / params['rampup']
        p = 1.0 - p
        return math.exp(-p * p * 5.0)

    def rampdown(epoch):
        p = (epoch - (params['num_epochs'] - params['rampdown'])) * 0.5
        return math.exp(-(p * p) / rampdown)

    def learning_rate_fn(global_step):
        epoch = global_step // params['num_epochs']
        ru = tf.cond(tf.less(epoch, params['rampup']),
                     lambda: rampup(epoch), lambda: tf.constant(1.0))
        rd = tf.cond(
            tf.greater_equal(epoch, params['num_epochs'] - params['rampdown']),
            lambda: rampdown(epoch), lambda: tf.constant(1.0))
        learning_rate = params['lr'] * ru * rd
        adam_beta =
        rd * params['adam_beta1'] + (1.0 - rd) params['rd_beta_target']
        unsup_wght = tf.cond(
            tf.equal(epoch, 0),
            lambda: ru * params['scale_unsup_wght_max'], lambda: 0.0)
        return learning_rate, adam_beta, unsup_wght

    return learning_rate_fn


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


class SSLDeepLensModel(resnet_model_ssl.ModelSSL):
    """ssl deep lens model use standard tf models"""

    def __init__(self, resnet_size=46, data_format=None, num_classes=2,
                 dtype=tf.float32):
        super(SSLDeepLensModel, self).__init__(
            resnet_size=resnet_size,
            bottleneck=True,
            num_classes=num_classes,
            num_filters=16,
            kernel_size=7,
            conv_stride=1,
            first_pool_size=None,
            first_pool_stride=None,
            block_sizes=[3, 3, 3, 3, 3],
            block_strides=[],
            resnet_version=2,
            data_format=data_format,
            dtype=dtype
        )


def ssl_deeplens_model_fn(features, labels, masks, targets, mode, model_class,
                          params=tempens_params):
    model = model_class()
    logits, embed = model(features, mode == tf.estimator.ModeKeys.TRAIN)
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            }
        )

    pred_t = tf.nn.softmax(logits)
    labeled_cost = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits) * masks
    )
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate_fn = params_update(params)
        learning_rate, adam_beta, unsup_wght = learning_rate_fn(global_step)
        unlabeled_cost = tf.reduce_mean(tf.square(pred_t - targets))
        cost = labeled_cost + unsup_wght * unlabeled_cost
        # TODO: add if embed
        half = tf.to_int32(tf.to_float(tf.shape(embed)[0]) / 2.)
        eucd2 = tf.reduce_mean(tf.square(embed[:half] - embed[half:]), axis=1)
        eucd = tf.sqrt(eucd2, name='eucd')
        margin = tf.constant(1.0, dtype=tf.float32, name='margin')
        target_hard = tf.to_int32(tf.argmax(targets, axis=1))
        merged_tar = tf.where(tf.equal(masks, 0), target_hard, labels)
        neighbor_bool = tf.equal(merged_tar[:half], merged_tar[half:])
        embed_loasses = tf.where(neighbor_bool, eucd2,
                                 tf.square(tf.maximum(margin - eucd, 0)))
        embed_loss = tf.reduce_mean(embed_loasses, name='loss')
        cost += unsup_wght * embed_loss * params['coeff_embed']

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=adam_beta)
        infer = optimizer.minimize(cost)
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(infer, update_op)
    else:
        train_op = None
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=cost,
        train_op=train_op
    )


def define_ssl_deeplens_flags():
    pass


def run_ssl_deeplens(flags_obj):
    pass


def main(_):
    with logger.benchmark_context(flags.FLAGS):
        run_ssl_deeplens(flags_obj)


if __name__ == "__main__":
    # tf.logging.set_verbosity(tf.logging.INFO)
    # define_ssl_deeplens_flags()
    # absl_app.run(main)
    load_dataset(_DB_PATH_1_)
