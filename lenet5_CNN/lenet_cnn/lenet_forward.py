import tensorflow as tf
IMAGE_SIZE = 32         # 图片大小
CHANNELS = 3            # 通道数
CONV1_SIZE = 5          # 第一层卷积核的大小
CONV1_KERNEL_NUM = 32   # 第一层卷积核的数量
CONV2_SIZE = 5          # 第二层卷积核的大小
CONV2_KERNEL_NUM = 64   # 第二层卷积核的数量
FIRST_SIZE = 256        # 第一层全连接层的神经元个数
OUTPUT_NODES = 2        # 输出层的神经元个数


def get_weight(shape, regularizer):          # 生成W参数W
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):            # 生成参数bias
    b = tf.Variable(tf.zeros(shape))        # 初始为零
    return b


def conv2d(x, kernel):            # 卷积层过滤
    # padding填充
    return tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')


def max_pool2d(x):      # 最大池化特征提取
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')      # strides is 2x2,use padding


def forward(x, train, regularizer):

    # 初始化第一卷积层
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, CHANNELS, CONV1_KERNEL_NUM], regularizer)
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    conv1 = conv2d(x, conv1_w)
    # 用relu激活函数
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 = max_pool2d(relu1)       # max pool

    # 初始化第二卷积层
    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool2d(relu2)

    pool_shape = pool2.get_shape().as_list()        # pool2维度
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]       # length*width*depth = 特征数量
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])      # reshape 行 = batchsize, 列 = nodes

    # 第一全连接层
    first_fc_w = get_weight([nodes, FIRST_SIZE], regularizer)
    first_fc_b = get_bias([FIRST_SIZE])
    first_fc = tf.nn.relu(tf.matmul(reshaped, first_fc_w)+first_fc_b)    # 经过全连接层的结果
    if train:
        first_fc = tf.nn.dropout(first_fc, 0.5)        # 第一层全连接层在训练时50%dropout

    # 第二层全连接层
    second_fc_w = get_weight([FIRST_SIZE, OUTPUT_NODES], regularizer)
    second_fc_b = get_bias([OUTPUT_NODES])
    y = tf.matmul(first_fc, second_fc_w)+second_fc_b     # 得到输出
    return y




















