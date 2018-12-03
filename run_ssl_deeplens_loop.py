import tensorflow as tf
from absl import app as absl_app
from absl import flags
import resnet_model_ssl
from official.resnet import resnet_run_loop
from official.utils.logs import logger


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


def ssl_deeplens_model_fn(features, labels, mode, model_class):
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

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec()


def define_ssl_deeplens_flags():
    pass


def run_ssl_deeplens(flags_obj):
    pass


def main(_):
    with logger.benchmark_context(flags.FLAGS):
        run_ssl_deeplens(flags_obj)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    define_ssl_deeplens_flags()
    absl_app.run(main)
