import os
import numpy as np
import tensorflow as tf


def get_file(file_dir):
    '''
    :param file_dir: 文件保存路径
    :return: 文件名， label
    '''
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []

    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'cat':
            cats.append(file_dir+file)
            label_cats.append(0)
        else:
            dogs.append(file_dir+file)
            label_dogs.append(1)
    print('there are %d cats\n there are %d dogs' % (len(cats), len(dogs)))

    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    temp = np.array([image_list, label_list])       # (2, 402)
    temp = temp.transpose()                         # (402, 2）
    print(np.shape(temp))
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(label) for label in label_list]  # 转换数据类型
    return image_list, label_list

def get_batch(image_list, label_list, image_h, image_w, batch_size, capacity):
    '''
    :param image:
    :param label:
    :param image_w:
    :param image_h:
    :param batch_size:
    :param capacity: 队列容量
    :return:
    '''
    image_list = tf.cast(image_list, tf.string)
    label_list = tf.cast(label_list, tf.int32)

    input_queue = tf.train.slice_input_producer([image_list, label_list])
    image = tf.read_file(input_queue[0])
    label = input_queue[1]
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image, image_h, image_w)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size,
                                              num_threads=64, capacity=capacity)
    return image_batch, label_batch



'''
train_dir = '/home/yxk/project/yxk_project_exercise/catordog/data/train_mini/'
a, b = get_file(train_dir)
c, d = get_batch(a, b, 200, 300, 5, 1000)
print(c, d)
'''

