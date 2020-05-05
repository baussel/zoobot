
import collections
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from zoobot.active_learning import create_instructions, run_estimator_config
from zoobot.estimators import losses, bayesian_estimator_funcs

import os
import math
import string
import collections
import glob
from functools import partial

from six.moves import xrange

from keras_applications.imagenet_utils import _obtain_input_shape

backend = tf.keras.backend
layers = tf.keras.layers
models = tf.keras.models
utils = tf.keras.utils
keras_utils = tf.keras.utils 

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

def get_swish():

    def swish(x):
        """Swish activation function: x * sigmoid(x).
        Reference: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
        """

        return tf.nn.swish(x)

    return swish

class PermaDropout(layers.Dropout):

    def call(self, x, training=None):
        return super().call(x, training=True)  # ME, force dropout on at test time


def get_dropout():
    """Wrapper over custom dropout. Fix problem of ``None`` shape for tf.keras.
    It is not possible to define FixedDropout class as global object,
    because we do not have modules for inheritance at first time.
    Issue:
        https://github.com/tensorflow/tensorflow/issues/30946
    """
    class FixedDropout(PermaDropout):  # inherit from mine instead of layers.Dropout
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = tf.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)



    return FixedDropout


def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier."""

    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""

    return int(math.ceil(depth_coefficient * repeats))


def mb_conv_block(inputs, block_args, activation, drop_rate=None, prefix='', ):
    """Mobile Inverted Residual Bottleneck.
    If drop_rate is specified, dropout is added to the end
    """

    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    bn_axis = 3  # channels last

    # workaround over non working dropout with None in noise_shape in tf.keras
    Dropout = get_dropout()

    # Expansion phase
    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:
        x = layers.Conv2D(filters, 1,
                          padding='same',
                          use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name=prefix + 'expand_conv')(inputs)
        x = layers.BatchNormalization(axis=bn_axis, name=prefix + 'expand_bn')(x)
        x = layers.Activation(activation, name=prefix + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    x = layers.DepthwiseConv2D(block_args.kernel_size,
                               strides=block_args.strides,
                               padding='same',
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=prefix + 'dwconv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=prefix + 'bn')(x)
    x = layers.Activation(activation, name=prefix + 'activation')(x)

    # Squeeze and Excitation phase
    if has_se:
        num_reduced_filters = max(1, int(
            block_args.input_filters * block_args.se_ratio
        ))
        se_tensor = layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)

        target_shape = (1, 1, filters) if backend.image_data_format() == 'channels_last' else (filters, 1, 1)
        se_tensor = layers.Reshape(target_shape, name=prefix + 'se_reshape')(se_tensor)
        se_tensor = layers.Conv2D(num_reduced_filters, 1,
                                  activation=activation,
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_reduce')(se_tensor)
        se_tensor = layers.Conv2D(filters, 1,
                                  activation='sigmoid',
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_expand')(se_tensor)
        if backend.backend() == 'theano':
            # For the Theano backend, we have to explicitly make
            # the excitation weights broadcastable.
            pattern = ([True, True, True, False] if backend.image_data_format() == 'channels_last'
                       else [True, False, True, True])
            se_tensor = layers.Lambda(
                lambda x: backend.pattern_broadcast(x, pattern),
                name=prefix + 'se_broadcast')(se_tensor)
        x = layers.multiply([x, se_tensor], name=prefix + 'se_excite')

    # Output phase
    x = layers.Conv2D(block_args.output_filters, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=prefix + 'project_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=prefix + 'project_bn')(x)
    if block_args.id_skip and all(
            s == 1 for s in block_args.strides
    ) and block_args.input_filters == block_args.output_filters:
        if drop_rate and (drop_rate > 0):
            x = Dropout(drop_rate,
                        noise_shape=(None, 1, 1, 1),
                        name=prefix + 'drop')(x)
        x = layers.add([x, inputs], name=prefix + 'add')

    return x

def EfficientNet(width_coefficient,
                 depth_coefficient,
                 default_resolution,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 model_name='efficientnet',
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 **kwargs):
    """Instantiates the EfficientNet architecture using given scaling coefficients.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_resolution: int, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: int.
        blocks_args: A list of BlockArgs to construct block modules.
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False.
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    global backend, layers, models, keras_utils

    # Determine proper input shape
    # default 224x224x3 if input_shape=none
    # https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_resolution,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if backend.backend() == 'tensorflow':
            from tensorflow.python.keras.backend import is_keras_tensor
        else:
            is_keras_tensor = backend.is_keras_tensor
        if not is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    activation = get_swish(**kwargs)

    # Build stem
    x = img_input
    x = layers.Conv2D(round_filters(32, width_coefficient, depth_divisor), 3,
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = layers.Activation(activation, name='stem_activation')(x)

    # Build blocks
    num_blocks_total = sum(block_args.num_repeat for block_args in blocks_args)
    block_num = 0
    for idx, block_args in enumerate(blocks_args):
        assert block_args.num_repeat > 0
        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(
            input_filters=round_filters(block_args.input_filters,
                                        width_coefficient, depth_divisor),
            output_filters=round_filters(block_args.output_filters,
                                         width_coefficient, depth_divisor),
            num_repeat=round_repeats(block_args.num_repeat, depth_coefficient))

        # The first block needs to take care of stride and filter size increase.
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = mb_conv_block(x, block_args,
                          activation=activation,
                          drop_rate=drop_rate,  # actually related to drop_connect_rate
                          prefix='block{}a_'.format(idx + 1))
        block_num += 1
        if block_args.num_repeat > 1:
            # pylint: disable=protected-access
            block_args = block_args._replace(
                input_filters=block_args.output_filters, strides=[1, 1])
            # pylint: enable=protected-access
            for bidx in xrange(block_args.num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                block_prefix = 'block{}{}_'.format(
                    idx + 1,
                    string.ascii_lowercase[bidx + 1]
                )
                x = mb_conv_block(x, block_args,
                                  activation=activation,
                                  drop_rate=drop_rate,
                                  prefix=block_prefix)
                block_num += 1

    # Build top
    x = layers.Conv2D(round_filters(1280, width_coefficient, depth_divisor), 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='top_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
    x = layers.Activation(activation, name='top_activation')(x)
    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate and dropout_rate > 0:
            # x = layers.Dropout(dropout_rate, name='top_dropout')(x)
            # use constantly-on dropout instead
            # top layer dropout needs to be high to do anything much
            x = PermaDropout(dropout_rate, name='top_dropout')(x)  
        x = layers.Dense(classes,
                         activation='softmax',
                         kernel_initializer=DENSE_KERNEL_INITIALIZER,
                         name='probs')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model
    model = models.Model(inputs, x, name=model_name)

    return model


def EfficientNetB0(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        1.0, 1.0, 224, 0.2,
        model_name='efficientnet-b0',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB3(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.3,  # added here  as it has an explicit default
                   **kwargs):
    return EfficientNet(
        1.2, 1.4, 300, dropout_rate,
        model_name='efficientnet-b3',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNet_custom_top(schema, input_shape=None, batch_size=16, add_channels=False, get_effnet=EfficientNetB0, **kwargs):

    output_dim = len(schema.answers)

    if add_channels:
      model = tf.keras.Sequential([tf.keras.layers.Lambda(lambda x: tf.stack([x, x, x], axis=3))])  # need channel dim for imagenet
    else:
      model = tf.keras.Sequential()
    # classes probably does nothing without include_top
    effnet = get_effnet(
        input_shape=input_shape,
        # input_tensor=tf.keras.Input(shape=input_shape, batch_size=batch_size),
        weights=None,
        include_top=False,
        classes=output_dim,
        **kwargs
    )
    model.add(effnet)

    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(output_dim))
    model.add(tf.keras.layers.Lambda(lambda x: tf.concat([tf.nn.softmax(x[:, q[0]:q[1]+1]) for q in schema.question_index_groups], axis=1), output_shape=[batch_size, output_dim]))        

    # will be updated by callback
    model.step = tf.Variable(0, dtype=tf.int64, name='model_step', trainable=False)

     # my loss only works with run_config shards, the new custom shards are vote fraction labelled
    loss_func = lambda x, y: losses.multiquestion_loss(x, y, question_index_groups=schema.question_index_groups)

    custom_mses = [bayesian_estimator_funcs.CustomMSEByColumn(name=q.text, start_col=start_col, end_col=end_col) for q, (start_col, end_col) in schema.named_index_groups.items()]
    model.compile(
        loss=loss_func,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=custom_mses
    )

    return model


if __name__ == '__main__':

    # https://github.com/tensorflow/tensorflow/issues/24496
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)

    version = 'gz2'

    shard_img_size = 128
    final_size = 128
    channels = 3

    warm_start = False
    test = False
    epochs = 1
    # shard_dir = f'data/decals/shards/multilabel_master_filtered_{shard_img_size}
    shard_dir = f'data/gz2/shards/all_featp5_facep5_sim_{shard_img_size}'

    save_dir = 'data/debug'

    batch_size = 4

    if version == 'decals':
        questions = [
            'smooth-or-featured',
            'has-spiral-arms',
            'bar',
            'bulge-size'
        ]
        label_cols = [
            'smooth-or-featured_smooth',
            'smooth-or-featured_featured-or-disk',
            'has-spiral-arms_yes',
            'has-spiral-arms_no',
            'bar_strong',
            'bar_weak',
            'bar_no',
            'bulge-size_dominant',
            'bulge-size_large',
            'bulge-size_moderate',
            'bulge-size_small',
            'bulge-size_none'
        ]

    elif version == 'gz2':
        questions = [
            'smooth-or-featured',
            'has-spiral-arms',
            'bar',
            'bulge-size'
        ]
        label_cols = [
            'smooth-or-featured_smooth',
            'smooth-or-featured_featured-or-disk',
            'has-spiral-arms_yes',
            'has-spiral-arms_no',
            'bar_yes',
            'bar_no',
            'bulge-size_dominant',
            'bulge-size_obvious',
            'bulge-size_just-noticeable',
            'bulge-size_no'
        ]

    schema = losses.Schema(label_cols, questions, version=version)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    train_records_dir = os.path.join(shard_dir, 'train_shards')
    eval_records_dir = os.path.join(shard_dir, 'eval_shards')
    train_records = [os.path.join(train_records_dir, x) for x in os.listdir(train_records_dir) if x.endswith('.tfrecord')]
    eval_records = [os.path.join(eval_records_dir, x) for x in os.listdir(eval_records_dir) if x.endswith('.tfrecord')]

    run_config = run_estimator_config.RunEstimatorConfig(
        initial_size=shard_img_size,
        final_size=final_size,
        schema=schema,
        epochs=epochs,  # to tweak 2000 for overnight at 8 iters, 650 for 2h per iter
        log_dir=save_dir,
        warm_start=warm_start,
        batch_size=batch_size
    )
    train_config = run_estimator_config.get_train_config(train_records, label_cols, batch_size, shard_img_size, final_size, channels)
    eval_config = run_estimator_config.get_eval_config(eval_records, label_cols, batch_size, shard_img_size, final_size, channels)
    # model = run_estimator_config.get_model(label_cols, questions, final_size, version=version)
    model = EfficientNet_custom_top(input_shape=(final_size, final_size, 1), schema=schema, batch_size=batch_size, add_channels=False)
    run_config.assemble(train_config, eval_config, model)

    trained_model_128 = run_config.run_estimator() 
