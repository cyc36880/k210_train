# Tensorflow mandates these.
from __future__ import absolute_import, division, print_function

from collections import namedtuple

import tensorflow as tf

slim = tf.contrib.slim

# Conv and DepthSepConv namedtuple define layers of the MobileNet architecture
# Conv defines 3x3 convolution layers
# DepthSepConv defines 3x3 depthwise convolution followed by 1x1 convolution.
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer
Conv = namedtuple("Conv", ["kernel", "stride", "depth"])
DepthSepConv = namedtuple("DepthSepConv", ["kernel", "stride", "depth"])

# MOBILENETV1_CONV_DEFS specifies the MobileNet body
MOBILENETV1_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024),
]

# MOBILENETV1_CONV_DEFS = [
#     Conv(kernel=[3, 3], stride=2, depth=32),
#     DepthSepConv(kernel=[3, 3], stride=1, depth=64),
#     DepthSepConv(kernel=[3, 3], stride=2, depth=128),
#     DepthSepConv(kernel=[3, 3], stride=1, depth=128),
#     DepthSepConv(kernel=[3, 3], stride=2, depth=256),
#     DepthSepConv(kernel=[3, 3], stride=1, depth=256),
#     DepthSepConv(kernel=[3, 3], stride=2, depth=512),
#     DepthSepConv(kernel=[3, 3], stride=1, depth=512),
#     DepthSepConv(kernel=[3, 3], stride=1, depth=512),
#     DepthSepConv(kernel=[3, 3], stride=1, depth=512),
#     DepthSepConv(kernel=[3, 3], stride=1, depth=512),
#     DepthSepConv(kernel=[3, 3], stride=1, depth=512)
# ]


def _fixed_padding(inputs, kernel_size, rate=1):
    """Pads the input along the spatial dimensions independently of input size.

    Pads the input such that if it was used in a convolution with 'VALID' padding,
    the output would have the same dimensions as if the unpadded input was used
    in a convolution with 'SAME' padding.

    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
      rate: An integer, rate for atrous convolution.

    Returns:
      output: A tensor of size [batch, height_out, width_out, channels] with the
        input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    kernel_size_effective = [
        kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
        kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
    ]
    pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
    pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
    pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg[0], pad_end[0]], [pad_beg[1], pad_end[1]], [0, 0]])
    return padded_inputs


def _fixed_padding8(inputs):
    """Pads the input along the spatial dimensions independently of input size.

    Pads the input such that if it was used in a convolution with 'VALID' padding,
    the output would have the same dimensions as if the unpadded input was used
    in a convolution with 'SAME' padding.

    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
      rate: An integer, rate for atrous convolution.

    Returns:
      output: A tensor of size [batch, height_out, width_out, channels] with the
        input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    padded_inputs = tf.pad(inputs, [[0, 0], [8, 8], [0, 0], [0, 0]])
    return padded_inputs


def mobilenet_v1_base(
    inputs,
    final_endpoint="Conv2d_13_pointwise",
    min_depth=8,
    depth_multiplier=1.0,
    conv_defs=None,
    output_stride=None,
    use_explicit_padding=False,
    scope=None,
):
    """Mobilenet v1.

    Constructs a Mobilenet v1 network from inputs to the given final endpoint.

    Args:
      inputs: a tensor of shape [batch_size, height, width, channels].
      final_endpoint: specifies the endpoint to construct the network up to. It
        can be one of ['Conv2d_0', 'Conv2d_1_pointwise', 'Conv2d_2_pointwise',
        'Conv2d_3_pointwise', 'Conv2d_4_pointwise', 'Conv2d_5'_pointwise,
        'Conv2d_6_pointwise', 'Conv2d_7_pointwise', 'Conv2d_8_pointwise',
        'Conv2d_9_pointwise', 'Conv2d_10_pointwise', 'Conv2d_11_pointwise',
        'Conv2d_12_pointwise', 'Conv2d_13_pointwise'].
      min_depth: Minimum depth value (number of channels) for all convolution ops.
        Enforced when depth_multiplier < 1, and not an active constraint when
        depth_multiplier >= 1.
      depth_multiplier: Float multiplier for the depth (number of channels)
        for all convolution ops. The value must be greater than zero. Typical
        usage will be to set this value in (0, 1) to reduce the number of
        parameters or computation cost of the model.
      conv_defs: A list of ConvDef namedtuples specifying the net architecture.
      output_stride: An integer that specifies the requested ratio of input to
        output spatial resolution. If not None, then we invoke atrous convolution
        if necessary to prevent the network from reducing the spatial resolution
        of the activation maps. Allowed values are 8 (accurate fully convolutional
        mode), 16 (fast fully convolutional mode), 32 (classification mode).
      use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
        inputs so that the output dimensions are the same as if 'SAME' padding
        were used.
      scope: Optional variable_scope.

    Returns:
      tensor_out: output tensor corresponding to the final_endpoint.
      end_points: a set of activations for external use, for example summaries or
                  losses.

    Raises:
      ValueError: if final_endpoint is not set to one of the predefined values,
                  or depth_multiplier <= 0, or the target output_stride is not
                  allowed.
    """
    # def depth(d): return max(int(d * depth_multiplier), min_depth)
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = {}

    # Used to find thinned depths for each layer.
    if depth_multiplier <= 0:
        raise ValueError("depth_multiplier is not greater than zero.")

    if conv_defs is None:
        conv_defs = MOBILENETV1_CONV_DEFS

    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError("Only allowed output_stride values are 8, 16, 32.")

    padding = "SAME"
    if use_explicit_padding:
        padding = "VALID"
    with tf.variable_scope(scope, "MobilenetV1", [inputs]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding=padding):
            # The current_stride variable keeps track of the output stride of the
            # activations, i.e., the running product of convolution strides up to the
            # current network layer. This allows us to invoke atrous convolution
            # whenever applying the next convolution would result in the activations
            # having output stride larger than the target output_stride.
            current_stride = 1

            # The atrous convolution rate parameter.
            rate = 1

            # net = inputs
            net = _fixed_padding8(inputs)
            for i, conv_def in enumerate(conv_defs):
                end_point_base = "Conv2d_%d" % i

                if output_stride is not None and current_stride == output_stride:
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    layer_stride = 1
                    layer_rate = rate
                    rate *= conv_def.stride
                else:
                    layer_stride = conv_def.stride
                    layer_rate = 1
                    current_stride *= conv_def.stride

                if isinstance(conv_def, Conv):
                    end_point = end_point_base
                    if use_explicit_padding:
                        net = _fixed_padding(net, conv_def.kernel)

                    if conv_def.stride == 2:
                        net = tf.space_to_batch(net, [[1, 1], [1, 1]], block_size=1, name=None)
                        net = slim.conv2d(
                            net,
                            depth(conv_def.depth),
                            conv_def.kernel,
                            stride=conv_def.stride,
                            normalizer_fn=slim.batch_norm,
                            padding="VALID",
                            scope=end_point,
                        )
                    else:
                        net = slim.conv2d(
                            net,
                            depth(conv_def.depth),
                            conv_def.kernel,
                            stride=conv_def.stride,
                            normalizer_fn=slim.batch_norm,
                            scope=end_point,
                        )
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points

                elif isinstance(conv_def, DepthSepConv):
                    end_point = end_point_base + "_depthwise"

                    if use_explicit_padding:
                        net = _fixed_padding(net, conv_def.kernel, layer_rate)

                    if layer_stride == 2:
                        net = tf.space_to_batch(net, [[1, 1], [1, 1]], block_size=1, name=None)
                        net = slim.separable_conv2d(
                            net,
                            None,
                            conv_def.kernel,
                            depth_multiplier=1,
                            stride=layer_stride,
                            rate=layer_rate,
                            padding="VALID",
                            normalizer_fn=slim.batch_norm,
                            scope=end_point,
                        )
                    else:

                        net = slim.separable_conv2d(
                            net,
                            None,
                            conv_def.kernel,
                            depth_multiplier=1,
                            stride=layer_stride,
                            rate=layer_rate,
                            normalizer_fn=slim.batch_norm,
                            scope=end_point,
                        )

                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points

                    end_point = end_point_base + "_pointwise"

                    net = slim.conv2d(
                        net, depth(conv_def.depth), [1, 1], stride=1, normalizer_fn=slim.batch_norm, scope=end_point
                    )

                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points
                else:
                    raise ValueError("Unknown convolution type %s for layer %d" % (conv_def.ltype, i))
    raise ValueError("Unknown final endpoint %s" % final_endpoint)


def inference(inputs, phase_train=True, yolo_conv_depth=125, weight_decay=0.0, depth_multiplier=1.0, reuse=None):
    batch_norm_params = {
        "decay": 0.995,
        "epsilon": 0.001,
        "scale": True,
        "is_training": phase_train,
        "updates_collections": tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
        [slim.conv2d, slim.separable_conv2d],
        weights_initializer=slim.initializers.xavier_initializer(),
        weights_regularizer=slim.l2_regularizer(weight_decay),
        normalizer_fn=slim.batch_norm,
        activation_fn=tf.nn.relu6,
        normalizer_params=batch_norm_params,
    ):
        return mobilenetv1_yolo_lite(
            inputs,
            is_training=phase_train,
            yolo_conv_depth=yolo_conv_depth,
            depth_multiplier=depth_multiplier,
            reuse=reuse,
        )


def mobilenetv1_yolo_lite(
    inputs,
    yolo_conv_depth=125,
    is_training=True,
    min_depth=8,
    depth_multiplier=1.0,
    conv_defs=None,
    reuse=None,
    scope="MobilenetV1",
):
    """Mobilenet v1 model for classification.

    Args:
      inputs: a tensor of shape [batch_size, height, width, channels].
      num_classes: number of predicted classes. If 0 or None, the logits layer
        is omitted and the input features to the logits layer (before dropout)
        are returned instead.
      dropout_keep_prob: the percentage of activation values that are retained.
      is_training: whether is training or not.
      min_depth: Minimum depth value (number of channels) for all convolution ops.
        Enforced when depth_multiplier < 1, and not an active constraint when
        depth_multiplier >= 1.
      depth_multiplier: Float multiplier for the depth (number of channels)
        for all convolution ops. The value must be greater than zero. Typical
        usage will be to set this value in (0, 1) to reduce the number of
        parameters or computation cost of the model.
      conv_defs: A list of ConvDef namedtuples specifying the net architecture.
      prediction_fn: a function to get predictions out of logits.
      spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
          of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
      global_pool: Optional boolean flag to control the avgpooling before the
        logits layer. If false or unset, pooling is done with a fixed window
        that reduces default-sized inputs to 1x1, while larger inputs lead to
        larger outputs. If true, any input size is pooled down to 1x1.

    Returns:
      net: a 2D Tensor with the logits (pre-softmax activations) if num_classes
        is a non-zero integer, or the non-dropped-out input to the logits layer
        if num_classes is 0 or None.
      end_points: a dictionary from components of the network to the corresponding
        activation.

    Raises:
      ValueError: Input rank is invalid.
    """
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError("Invalid input tensor rank, expected 4, was: %d" % len(input_shape))

    with tf.variable_scope(scope, "MobilenetV1", [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = mobilenet_v1_base(
                inputs, scope=scope, min_depth=min_depth, depth_multiplier=depth_multiplier, conv_defs=conv_defs
            )
        with tf.variable_scope("detect_layer"):
            net = slim.separable_conv2d(
                net, None, [3, 3], depth_multiplier=1, stride=1, padding="SAME", normalizer_fn=slim.batch_norm
            )

            net = slim.conv2d(net, 1024, [1, 1], stride=1, normalizer_fn=slim.batch_norm)

            with tf.variable_scope("yolo_out"):
                net = slim.conv2d(
                    net, yolo_conv_depth, [1, 1], stride=1, padding="SAME", activation_fn=None, normalizer_fn=None
                )

    return net, None
