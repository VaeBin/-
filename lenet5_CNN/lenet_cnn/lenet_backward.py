import lenet_forward
import tensorflow as tf
import os
import numpy as np
import PIL.Image
import time

BATCH_SIZE = 100                         # 批处理大小
REGULARIZER = 0.0015                     # 正则化参数
LEARNING_RATE_BASE = 0.001               # 学习率即衰减率
LEARNING_RATE_DACAY = 0.99
MOVING_AVERAGE_DECAY = 0.99              # 滑动平均衰减
STEPS = 3500                             # 训练次数
MODEL_SAVE_PATH = "./model/"             # 模型保存相对路径
MODEL_NAME = "lenet_model"               # 模型保存文件名
DECAY_TIMES = 50                         # 学习率衰减频次
TRAIN_NORMAL_PATH = "./images/train/normal/"              # 训练集路径
TRAIN_RESTRICTED_PATH = "./images/train/restricted/"


def backward():
    # 输入占位
    x = tf.placeholder(tf.float32,[
        BATCH_SIZE,
        lenet_forward.IMAGE_SIZE,
        lenet_forward.IMAGE_SIZE,
        lenet_forward.CHANNELS
    ])
    y_ = tf.placeholder(tf.float32, [None, lenet_forward.OUTPUT_NODES])     # 标签占位
    y = lenet_forward.forward(x, True, REGULARIZER)     # 使用dropout
    global_step = tf.Variable(0, trainable=False)       # 训练次数

    # 交叉熵损失函数
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))          # 一批一批损失计算

    # 自适应学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        DECAY_TIMES,
        LEARNING_RATE_DACAY,
        staircase=True
    )

    # 从梯度下降到Adam优化算法
    # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

    # 滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    # 创建saver存储模型
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 全局变量初始化
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 训练集中正常的图片所在的文件夹目录
        dirs1 = os.listdir(TRAIN_NORMAL_PATH)
        # 训练集中有缺陷的图片所在的文件夹目录
        dirs2 = os.listdir(TRAIN_RESTRICTED_PATH)

        # 将第一张正常的图片存进array1
        first_img_path = TRAIN_NORMAL_PATH + dirs1[0]
        image = PIL.Image.open(first_img_path)
        re_img = image.resize((lenet_forward.IMAGE_SIZE, lenet_forward.IMAGE_SIZE),PIL.Image.ANTIALIAS)
        re_arr = np.array(re_img)
        re_arr = re_arr / 255
        array1 = np.array([re_arr])

        # 将第一张有缺陷的图片存进array2
        second_img_path = TRAIN_RESTRICTED_PATH + dirs2[0]
        image = PIL.Image.open(second_img_path)
        re_img = image.resize((lenet_forward.IMAGE_SIZE, lenet_forward.IMAGE_SIZE),PIL.Image.ANTIALIAS)
        re_arr = np.array(re_img)
        re_arr = re_arr / 255
        array2 = np.array([re_arr])

        # 将所有的正常图片存进array1
        for j in range(1, len(dirs1)):
            path1 = TRAIN_NORMAL_PATH + dirs1[j]         # 得到每张图片路径
            image = PIL.Image.open(path1)        # 打开图片
            reshaped_img = image.resize((lenet_forward.IMAGE_SIZE, lenet_forward.IMAGE_SIZE),PIL.Image.ANTIALIAS)
            reshaped_arr = np.array(reshaped_img)
            reshaped_arr = reshaped_arr / 255
            dim = array1.shape        # 数组维数
            array1 = np.append(array1, reshaped_arr)        # 将图片加到array1数组
            array1 = array1.reshape(dim[0] + 1, dim[1], dim[2], dim[3])          # 调整reshape array1

        # 同上，对于有缺陷的图片存进array2
        for j in range(1, len(dirs2)):
            path2 = TRAIN_RESTRICTED_PATH + dirs2[j]
            image = PIL.Image.open(path2)
            reshaped_img = image.resize((lenet_forward.IMAGE_SIZE, lenet_forward.IMAGE_SIZE),PIL.Image.ANTIALIAS)
            reshaped_arr = np.array(reshaped_img)
            reshaped_arr = reshaped_arr / 255
            dim = array2.shape
            array2 = np.append(array2, reshaped_arr)
            array2 = array2.reshape(dim[0] + 1, dim[1], dim[2], dim[3])

        for i in range(STEPS):
            for j in range(50):      # 训练集可以分成batch_size 为100，共有500有缺陷，1500正常，所以循环50batch就是训练完所有训练集一次
                start = (j * 25) % 500
                end = (j * 25) % 500 + (int)(BATCH_SIZE/4)

                l1 = [1, 0]*75        # 一个batch中正常图片的标签75个
                l2 = [0, 1]*25        # 一个batch中有缺陷图片的标签25个
                ll = l1 + l2          # 一个batch中图片的标签100个
                ys = np.array(ll)
                ys = ys.reshape((100, 2))

                xs = array1[start*3:end*3, :, :, :]     # 75个正常图片的读入
                xs_dim = xs.shape
                xs = np.append(xs, array2[start:end, :, :, :])        # 加上25个非正常的图片的读入
                xs = np.reshape(xs, (xs_dim[0] + 25, xs_dim[1], xs_dim[2], xs_dim[3]))

            # 训练、计算损失、训练次数
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 20 == 0:           # 每20次训练就打印一次训练步数和损失
                print("after %d training step(s),loss on training batch is %g." % (step, loss_value))
            if i % 100 == 0:
                # 每100次训练就保存一次训练模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


if __name__ == '__main__':
    backward()










