import time
import tensorflow as tf
import sys
import numpy as np
import lenet_forward
import lenet_backward
import PIL.Image
import os
BATCH_SIZE = 750

VALIDATE_IMAGE_NORMAL_PATH = "./images/validate/normal/"
VALIDATE_IMAGE_RESTRICTED_PATH = "./images/validate/restricted/"
LENET_VALIDATE_NUM_EXAMPLES = 750


def test():
    # 计算图
    with tf.Graph().as_default() as g:
        # 占位x
        x = tf.placeholder(tf.float32, [
            LENET_VALIDATE_NUM_EXAMPLES,
            lenet_forward.IMAGE_SIZE,
            lenet_forward.IMAGE_SIZE,
            lenet_forward.CHANNELS
        ])
        # 占位标签
        y_ = tf.placeholder(tf.float32, [None, lenet_forward.OUTPUT_NODES])
        # 占位输出
        y = lenet_forward.forward(x, False, None)           # 验证的时候不使用dropout

        # 滑动平均
        ema = tf.train.ExponentialMovingAverage(lenet_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)         # 保存滑动平均的值


        # 准确率
        correct_probability = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_probability, tf.float32))       # 转化为float32型数据

        with tf.Session() as sess:
            # 加载上次训练的已有模型进行验证
            ckpt = tf.train.get_checkpoint_state(lenet_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                # 得到训练次数
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split(' ')[-1]

                # 找到验证集的两个图片文件目录：normal的、restricted的
                dirs1 = os.listdir(VALIDATE_IMAGE_NORMAL_PATH)
                dirs2 = os.listdir(VALIDATE_IMAGE_RESTRICTED_PATH)

                # 将正常的第一张图片读入array1
                first_img_path = VALIDATE_IMAGE_NORMAL_PATH + dirs1[0]
                image = PIL.Image.open(first_img_path)
                re_img = image.resize((lenet_forward.IMAGE_SIZE, lenet_forward.IMAGE_SIZE),PIL.Image.ANTIALIAS)
                re_arr = np.array(re_img)
                re_arr = re_arr/255
                array1 = np.array([re_arr])

                # 将非正常的第一张图片读入array2
                second_img_path = VALIDATE_IMAGE_RESTRICTED_PATH + dirs2[0]
                image = PIL.Image.open(second_img_path)
                re_img = image.resize((lenet_forward.IMAGE_SIZE, lenet_forward.IMAGE_SIZE),PIL.Image.ANTIALIAS)
                re_arr = np.array(re_img)
                re_arr = re_arr / 255
                array2 = np.array([re_arr])

                # 接着将验证集中normal的文件夹下所有的正常图片读进array1中
                for j in range(1, len(dirs1)):
                    path1 = VALIDATE_IMAGE_NORMAL_PATH + dirs1[j]  # 图片路径
                    image = PIL.Image.open(path1)  # 打开图片
                    reshaped_img = image.resize(
                        (lenet_forward.IMAGE_SIZE, lenet_forward.IMAGE_SIZE),PIL.Image.ANTIALIAS)
                    reshaped_arr = np.array(reshaped_img)
                    reshaped_arr = reshaped_arr / 255     # 将RGB转为（0，1）
                    dim = array1.shape          # 数组维数
                    array1 = np.append(array1, reshaped_arr)         # 将图片加到array1数组
                    array1 = array1.reshape(dim[0] + 1, dim[1], dim[2], dim[3])       # 重塑array1维数

                # 接着将验证集中normal的文件夹下所有的正常图片读进array1中
                for j in range(1, len(dirs2)):
                    path2 = VALIDATE_IMAGE_RESTRICTED_PATH + dirs2[j]           # 图片路径
                    image = PIL.Image.open(path2)           # 打开图片
                    reshaped_img = image.resize(
                        (lenet_forward.IMAGE_SIZE, lenet_forward.IMAGE_SIZE),PIL.Image.ANTIALIAS)
                    reshaped_arr = np.array(reshaped_img)
                    reshaped_arr = reshaped_arr/255         # 将RGB转化为（0，1）
                    dim = array2.shape          # 数组维度
                    array2 = np.append(array2, reshaped_arr)           # 将图片加到array2中
                    array2 = array2.reshape(dim[0] + 1, dim[1], dim[2], dim[3])          # 重塑array2维度


                l1 = [1, 0] * 500          # 验证集中所有的正常图片的标签，共500个标签
                l2 = [0, 1] * 250          # 验证集中所有的非正常图片的标签，共250个标签
                ll = l1 + l2               # 上述两种标签连在一起，共750个标签
                ys = np.array(ll)          # 处理成数组
                validate_ys = ys.reshape((750, 2))          # 重塑ys

                xs = array1[0:500, :, :, :]        # 验证集种500张正常图片
                xs_dim = xs.shape            # 获取array1的维度
                xs = np.append(xs, array2[0:250, :, :, :])          # 添加验证集种非正常图片250张
                reshaped_xs = np.reshape(xs, (xs_dim[0] + 250, xs_dim[1], xs_dim[2], xs_dim[3]))    # 重塑xs

                # 计算准确率，用字典喂数据
                accuracy_score = sess.run(accuracy, feed_dict={x: reshaped_xs, y_: validate_ys})
                print("after %s trainng step(s),test accuracy is %g" % (global_step, accuracy_score))
                print("测试完成")
            else:         # 没有找到现有模型
                print("no checkpoint is found")
                return

if __name__ == '__main__':
    test()






