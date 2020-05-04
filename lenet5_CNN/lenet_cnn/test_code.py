import sys
import os
import csv
import tensorflow as tf
import numpy as np
from PIL import Image
import lenet_backward
import lenet_forward
BATCH_SIZE = 1


def test(filepath):
    # 对输入占位
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        lenet_forward.IMAGE_SIZE,
        lenet_forward.IMAGE_SIZE,
        lenet_forward.CHANNELS
    ])
    # 对输出占位
    y = lenet_forward.forward(x, False, None)           # 测试不使用dropout
    global_step = tf.Variable(0, trainable=False)
    index = tf.argmax(y,1)             # 测试输出中得分最高的索引，【0，1】则为1，【1，0】则为0

    ema = tf.train.ExponentialMovingAverage(lenet_backward.MOVING_AVERAGE_DECAY)        # 滑动平均
    ema_restore = ema.variables_to_restore()
    saver = tf.train.Saver(ema_restore)

    with tf.Session() as sess:
        dir = os.listdir(filepath)            # 需要测试的图片文件目录
        ckpt = tf.train.get_checkpoint_state(lenet_backward.MODEL_SAVE_PATH)           # 加载训练过的已有模型
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        out = open('result.csv', 'w', newline='')             # 打开或者创建result.csv文件，写入方式
        csv_write = csv.writer(out,dialect='excel')           # 写表格形式到csv文件
        out_arr = ['pictures', 'if restricted?']              # 第一行表数据的提示行
        csv_write.writerow(out_arr)                           # 写入第一行提示行

        # 处理每一个图片文件
        for file in dir:
            path = filepath + file            # 得到一个图片文件的路径名
            image = Image.open(path)             #打开一张图片
            re_img = image.resize((lenet_forward.IMAGE_SIZE, lenet_forward.IMAGE_SIZE), Image.ANTIALIAS)     # 高质量处理图片
            re_arr = np.array(re_img)           # 图片转化成数组形式
            re_arr = re_arr / 255            # RGB化为（0，1）
            array1 = np.array([re_arr])
            index_1 = sess.run(index, feed_dict={x: array1})            # 计算索引
            index_2 = index_1[0]             # 目前索引是数组，我们需要得到0位
            out_arr = [file, index_2]            #形成临时数组，构成一行，第一列是图片名称，第二列是是否为非正常的行李箱
            csv_write.writerow(out_arr)             # 将一行表格数据写入csv文件
        print("write over")             # 写完

    print('测试完成')           # 测试完成

if __name__ == '__main__':
    tmp = sys.argv           # 终端参数
    if len(tmp) < 2:
        print('输入测试路径名')
    else:
        filepath = tmp[1]             # filepath为测试路径名（该路径名下有很多图片）
        if filepath[-1] != '/':           #如果文件夹结尾未加‘/’，则自动加上，因为后面的图片名首没有‘/’
            filepath = filepath + '/'
        print(filepath)               # 打印文件目录名
        test(filepath)                # 测试文件目录