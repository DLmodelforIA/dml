import tensorflow as tf


def _resize_3D(input_layer, scale_factor):
    '''
    '''
    _method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    shape = input_layer.get_shape()
    x = tf.reshape(input_layer, [-1, shape[1], shape[2], shape[3] * shape[4]])
    x = tf.image.resize_images(x, [shape[1] * scale_factor[0], shape[2] * scale_factor[1]], _method)
    x = tf.reshape(x, [-1, shape[1] * scale_factor[0], shape[2] * scale_factor[1], shape[3], shape[4]])
    x = tf.transpose(x, [0, 3, 2, 1, 4])
    x = tf.reshape(x, [-1, shape[3], shape[2] * scale_factor[1], shape[1] * scale_factor[0] * shape[4]])
    x = tf.image.resize_images(x, [shape[3] * scale_factor[2], shape[2] * scale_factor[1]], _method)
    x = tf.reshape(x,
                   [-1, shape[3] * scale_factor[2], shape[2] * scale_factor[1], shape[1] * scale_factor[0], shape[4]])
    x = tf.transpose(x, [0, 3, 2, 1, 4])
    return x


def _norm_func(x, norm):
    if norm == 'BN':
        return tf.layers.batch_normalization(x)
    elif norm == 'GN':
        return tf.contrib.layers.group_norm(x, groups=4, reduction_axes=(-4, -3, -2))
    else:
        raise NameError('Undifined normalization type')


def _activ_func(x, acticv_type):
    if acticv_type == 'Relu':
        return tf.nn.relu(x)
    elif acticv_type == 'LeakyRelu':
        return tf.nn.leaky_relu(x)
    else:
        raise NameError('Un-defined Activation Function')


def drop_out(x, rate=0.2, is_training=True, scope='dropout'):
    with tf.variable_scope(scope):
        x = tf.layers.dropout(x, rate=rate, training=is_training)
        return x


'''3d upsample block'''


def upsample3D_block(x, channels, norm='GN', activ_func='Relu', \
                     upsampleDims='ALL', upsampleFactor=None):
    '''
    '''
    if upsampleDims == 'ALL':
        x = _resize_3D(x, scale_factor=[2, 2, 2])
    elif upsampleDims == 'NoDepth':
        x = _resize_3D(x, scale_factor=[1, 2, 2])
    elif upsampleDims == 'USE_INPUT':
        x = _resize_3D(x, scale_factor=upsampleFactor)
    else:
        raise NameError('Undifined upsample type')
    x = tf.layers.conv3d(x, filters=channels, strides=1, kernel_size=3, padding='SAME')
    x = _norm_func(x, norm)
    x = _activ_func(x, acticv_type=activ_func)
    return x


'''3d conv-transpose block'''


def conv3D_T_block(x, channels, norm='GN', activ_func='Relu', upsampleDims='ALL'):
    '''
    '''
    if upsampleDims == 'ALL':
        x = tf.layers.conv3d_transpose(x, filters=channels, strides=2, kernel_size=4, padding='SAME')
    elif upsampleDims == 'NoDepth':
        x = tf.layers.conv3d_transpose(x, filters=channels, strides=[1, 2, 2], kernel_size=4, padding='SAME')
    else:
        raise NameError('Undifined upsample type')

    x = _norm_func(x, norm)
    x = _activ_func(x, acticv_type=activ_func)
    return x


'''3d conv block'''


def conv3D_block(x, channels, norm='GN', activ_func='Relu'):
    x = tf.layers.conv3d(x, filters=channels, strides=1, kernel_size=3, padding='SAME')
    # Cannot use '_norm_func' here due to it use 'tf.contrib.layers.batch_norm' 
    # instead of 'tf.layers.batch_normalization' in '_norm_func' ,
    # which is all the same but has a different 'name' of tensor.
    # This might casue name error when reload old-version ckpts.
    # can change to '_norm_func' in future when donot need load old-version ckpts
    if norm == 'GN':
        x = tf.contrib.layers.group_norm(x, groups=4, reduction_axes=(-4, -3, -2))
    elif norm == 'BN':
        x = tf.contrib.layers.batch_norm(x)
    x = _activ_func(x, acticv_type=activ_func)
    return x


def _GAP(x):
    return tf.reduce_mean(x, axis=[1, 2, 3])


'''SE block'''


def SE_block_3d(x, ratio=8, activ_func='Relu'):
    w = _GAP(x)
    channels = w.get_shape()[-1]
    if ratio > channels:
        ratio = channels
    w = tf.layers.dense(w, units=channels // ratio, )
    w = _activ_func(w, acticv_type=activ_func)
    w = tf.layers.dense(w, units=channels, )
    w = tf.nn.sigmoid(w)
    w = tf.reshape(w, [-1, 1, 1, 1, channels])
    return x * w


'''SE-Conv block'''


def conv3D_SE_block(x, channels, norm='GN', activ_func='Relu'):
    x = conv3D_block(x, channels, norm=norm, activ_func=activ_func)
    x = SE_block_3d(x, ratio=8, activ_func=activ_func)
    return x


'''ResX block'''


def _group_conv3d(x, groups=16, channels=128, size=3):
    subchannel = channels / groups
    input_list = tf.split(x, num_or_size_splits=groups, axis=-1)
    output_list = []
    for x in input_list:
        output_list.append( \
            tf.layers.conv3d(x, filters=subchannel, kernel_size=size, strides=1, padding='SAME')
        )
    res = tf.concat(output_list, axis=-1)
    return res


def ResX_block_3d(inputs, mid_channels=None, size=3, groups=32, norm='BN', activ_func='Relu'):
    channels = int(inputs.get_shape()[-1])
    if not mid_channels:
        mid_channels = channels // 2
    if mid_channels < 4:
        norm = 'BN'
    x = tf.layers.conv3d( \
        inputs=inputs, filters=mid_channels, kernel_size=1, strides=1, padding='SAME')
    x = _norm_func(x, norm)
    x = _activ_func(x, acticv_type=activ_func)
    x = _group_conv3d(x, groups=groups, channels=mid_channels, size=size)
    x = _norm_func(x, norm)
    x = tf.layers.conv3d( \
        inputs=x, filters=channels, kernel_size=1, strides=1, padding='SAME')
    x = _norm_func(x, norm)
    x = inputs + x
    x = _activ_func(x, acticv_type=activ_func)
    return x


'''SE-ResX block'''


def SE_ResX_block_3d(x, mid_channels=None, opt_channels=None, \
                     groups=16, ratio=16, norm='GN', activ_func='Relu'):
    x = ResX_block_3d( \
        x, mid_channels=mid_channels, groups=groups, norm=norm, activ_func=activ_func)
    x = SE_block_3d(x, ratio=ratio, activ_func=activ_func)
    return x


def vessel_attention(x, base_channel, va_channel=8, is_training=True, downsample_times='3',
                     reuse=tf.AUTO_REUSE, activ_func='Relu', upsample_type='resize', analysis_mode=False):
    '''
    Input:
        x[tensor]->[batch size, d, w, h, 1]
    Output:
        
    '''

    channels = base_channel

    def _upsample(x, channels, norm='GN', activ_func=activ_func, upsampleDims=None):
        if upsample_type == 'resize':
            return upsample3D_block(x, channels, norm=norm, activ_func=activ_func, upsampleDims=upsampleDims)
        elif upsample_type == 'deConv':
            return conv3D_T_block(x, channels, norm=norm, activ_func=activ_func, upsampleDims=upsampleDims)
        else:
            raise NameError('Un-defined Upsample Type')

    def _va_attention(x):
        x = tf.layers.conv3d(x, filters=1, strides=1, kernel_size=1, padding='SAME')
        x = tf.sigmoid(x)
        return x

    with tf.variable_scope("network", reuse=reuse):
        x = tf.identity(x, name='network_input')
        with tf.variable_scope("Down1", reuse=reuse):
            D1 = conv3D_SE_block(x, channels, norm='GN')
            D1 = SE_ResX_block_3d(D1, groups=channels // 4, norm='GN')
        with tf.variable_scope("Down2", reuse=reuse):
            D2 = tf.layers.max_pooling3d(D1, pool_size=[2, 2, 2], strides=[2, 2, 2])
            D2 = conv3D_SE_block(D2, channels * 2, norm='GN')
            D2 = SE_ResX_block_3d(D2, groups=channels // 2, norm='GN')
        with tf.variable_scope("Down3", reuse=reuse):
            D3 = tf.layers.max_pooling3d(D2, pool_size=[1, 2, 2], strides=[1, 2, 2])
            D3 = conv3D_SE_block(D3, channels * 4, norm='GN')
            D3 = SE_ResX_block_3d(D3, groups=channels, norm='GN')
            D3 = SE_ResX_block_3d(D3, groups=channels, norm='GN')
            D3 = SE_ResX_block_3d(D3, groups=channels, norm='GN')
        with tf.variable_scope("Middle", reuse=reuse):
            if downsample_times == '3':
                MD = tf.layers.max_pooling3d(D3, pool_size=[1, 2, 2], strides=[1, 2, 2])
            elif downsample_times == '2':
                MD = D3
            MD = conv3D_block(MD, channels * 8, norm='GN')
            MD = conv3D_block(MD, channels * 8, norm='GN')

        with tf.variable_scope("Vas_Att", reuse=reuse):
            if downsample_times == '3':
                V3 = _upsample(MD, va_channel, upsampleDims='NoDepth')
            elif downsample_times == '2':
                V3 = MD
            D3_V = conv3D_block(D3, va_channel, norm='GN')
            V3 = tf.concat([V3, D3_V], axis=-1)
            V3 = conv3D_block(V3, va_channel, norm='GN')
            V2 = _upsample(V3, va_channel, upsampleDims='NoDepth')
            D2_V = conv3D_block(D2, va_channel, norm='GN')
            V2 = tf.concat([V2, D2_V], axis=-1)
            V2 = conv3D_block(V2, va_channel, norm='GN')
            V1 = _upsample(V2, va_channel, upsampleDims='ALL')
            D1_V = conv3D_block(D1, va_channel, norm='GN')
            V1 = tf.concat([V1, D1_V], axis=-1)
            V1 = conv3D_block(V1, va_channel, norm='GN')
            V_OP = conv3D_block(V1, 2, norm='BN')
            V_OP = tf.nn.softmax(V_OP, axis=-1)  # **OP**
            VA1 = conv3D_block(V1, va_channel, norm='GN')
            VA1_att = _va_attention(VA1)
            VA2 = tf.layers.max_pooling3d(VA1, pool_size=[2, 2, 2], strides=[2, 2, 2])
            VA2 = conv3D_block(VA2, va_channel, norm='GN')
            VA2_att = _va_attention(VA2)
            VA3 = tf.layers.max_pooling3d(VA2, pool_size=[1, 2, 2], strides=[1, 2, 2])
            VA3 = conv3D_block(VA3, va_channel, norm='GN')
            VA3_att = _va_attention(VA3)

        with tf.variable_scope("Up3", reuse=reuse):
            if downsample_times == '3':
                U3 = _upsample(MD, channels * 4, upsampleDims='NoDepth')
            elif downsample_times == '2':
                U3 = MD
            U3 = tf.concat([U3, D3], axis=-1) * VA3_att
            U3 = conv3D_block(U3, channels * 4, norm='GN')
            U3 = drop_out(U3, rate=0.2, is_training=is_training, )
            U3 = conv3D_block(U3, channels * 4, norm='GN')
        with tf.variable_scope("Up2", reuse=reuse):
            U2 = _upsample(U3, channels * 2, upsampleDims='NoDepth')
            U2 = tf.concat([U2, D2], axis=-1) * VA2_att
            U2 = conv3D_block(U2, channels * 2, norm='GN')
            U2 = drop_out(U2, rate=0.2, is_training=is_training, )
            U2 = conv3D_block(U2, channels * 2, norm='GN')
        with tf.variable_scope("Up1", reuse=reuse):
            U1 = _upsample(U2, channels, upsampleDims='ALL')
            U1 = tf.concat([U1, D1], axis=-1) * VA1_att
            U1 = conv3D_block(U1, channels, norm='GN')
            U1 = drop_out(U1, rate=0.2, is_training=is_training, )
            U1 = conv3D_block(U1, channels, norm='GN')

        with tf.variable_scope("OP", reuse=reuse):
            OP = conv3D_block(U1, 2, norm='BN')
            OP = tf.nn.softmax(OP, axis=-1)

        OP = tf.identity(OP, name='network_output')

        if not analysis_mode:
            return [OP, V_OP]
        else:
            return [OP, V_OP], VA1_att
