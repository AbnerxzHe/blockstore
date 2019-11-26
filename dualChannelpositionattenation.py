from Vnet.layer import (conv3d, deconv3d, normalizationlayer, crop_and_concat, resnet_Add, upsample3d,
                        weight_xavier_init, bias_variable, save_images)
import tensorflow as tf
import numpy as np
import os


def positionAttentionblock(x, inputfilters, outfilters, kernal_size=1, scope=None):
    """
    Position attention module
    :param x:
    :param inputfilters:inputfilter number
    :param outfilters:outputfilter number
    :param scope:
    :return:
    """
    with tf.name_scope(scope):
        m_batchsize, Z, H, W, C = x.get_shape().as_list()

        kernalquery = (kernal_size, kernal_size, kernal_size, inputfilters, outfilters)
        Wquery = weight_xavier_init(shape=kernalquery,
                                    n_inputs=kernalquery[0] * kernalquery[1] * kernalquery[2] * kernalquery[3],
                                    n_outputs=kernalquery[-1], activefunction='relu',
                                    variable_name=scope + 'conv_Wquery')
        Bquery = bias_variable([kernalquery[-1]], variable_name=scope + 'conv_Bquery')
        query_conv = conv3d(x, Wquery) + Bquery
        query_conv_new = tf.reshape(query_conv, [-1, Z * H * W])

        kernalkey = (kernal_size, kernal_size, kernal_size, inputfilters, outfilters)
        Wkey = weight_xavier_init(shape=kernalkey, n_inputs=kernalkey[0] * kernalkey[1] * kernalkey[2] * kernalkey[3],
                                  n_outputs=kernalkey[-1], activefunction='relu', variable_name=scope + 'conv_Wkey')
        Bkey = bias_variable([kernalkey[-1]], variable_name=scope + 'conv_Bkey')
        key_conv = conv3d(x, Wkey) + Bkey
        key_conv_new = tf.reshape(key_conv, [-1, Z * H * W])

        # OOM,such as 512x512x32 then matric is 8388608x8388608
        # key_conv_new = tf.transpose(key_conv_new, [0, 2, 1])
        # (2,2,2,3)*(2,2,3,4)=(2,2,2,4),(2,2,3)*(2,3,4)=(2,2,4)
        # energy = tf.matmul(query_conv_new, key_conv_new)  # (m_batchsize,Z*H*W,Z*H*W)

        energy = tf.multiply(query_conv_new, key_conv_new)
        attention = tf.nn.sigmoid(energy)

        kernalproj = (kernal_size, kernal_size, kernal_size, inputfilters, outfilters)
        Wproj = weight_xavier_init(shape=kernalproj,
                                   n_inputs=kernalproj[0] * kernalproj[1] * kernalproj[2] * kernalproj[3],
                                   n_outputs=kernalproj[-1], activefunction='relu', variable_name=scope + 'conv_Wproj')
        Bproj = bias_variable([kernalproj[-1]], variable_name=scope + 'conv_Bproj')
        proj_value = conv3d(x, Wproj) + Bproj
        proj_value_new = tf.reshape(proj_value, [-1, Z * H * W])

        out = tf.multiply(attention, proj_value_new)
        out_new = tf.reshape(out, [-1, Z, H, W, C])

        out_new = resnet_Add(out_new, x)
        return out_new


def channelAttentionblock(x, scope=None):
    """
    Channel attention module
    :param x:input
    :param scope: scope name
    :return:channelattention result
    """
    with tf.name_scope(scope):
        m_batchsize, Z, H, W, C = x.get_shape().as_list()

        proj_query = tf.reshape(x, [-1, C])
        proj_key = tf.reshape(x, [-1, C])
        proj_query = tf.transpose(proj_query, [1, 0])

        energy = tf.matmul(proj_query, proj_key)  # (C,C)
        attention = tf.nn.sigmoid(energy)

        proj_value = tf.reshape(x, [-1, C])
        proj_value = tf.transpose(proj_value, [1, 0])
        out = tf.matmul(attention, proj_value)  # (C,-1)

        out = tf.reshape(out, [-1, Z, H, W, C])
        out = resnet_Add(out, x)
        return out


def conv_bn_relu_drop(x, kernal, phase, drop, image_z=None, height=None, width=None, scope=None):
    """
    :param x:
    :param kernal:
    :param phase:
    :param drop:
    :param image_z:
    :param height:
    :param width:
    :param scope:
    :return:
    """
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
        conv = conv3d(x, W) + B
        conv = normalizationlayer(conv, is_train=phase, height=height, width=width, image_z=image_z, norm_type='group',
                                  scope=scope)
        conv = tf.nn.dropout(tf.nn.relu(conv), drop)
        return conv


def down_sampling(x, kernal, phase, drop, image_z=None, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1],
                               activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv3d(x, W, 2) + B
        conv = normalizationlayer(conv, is_train=phase, height=height, width=width, image_z=image_z, norm_type='group',
                                  scope=scope)
        conv = tf.nn.dropout(tf.nn.relu(conv), drop)
        return conv


def deconv_relu(x, kernal, samefeture=False, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[-1],
                               n_outputs=kernal[-2], activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-2]], variable_name=scope + 'B')
        conv = deconv3d(x, W, samefeture, True) + B
        conv = tf.nn.relu(conv)
        return conv
