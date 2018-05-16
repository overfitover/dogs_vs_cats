import input_data
from model import model_fcn
import tensorflow as tf
import numpy as np
import os


TRAIN_DIR = './train_mini/'
LOGS_DIR = './logs_dir'
IMG_H = 208
IMG_W = 208
BATCH_SIZE = 16
CAPACITY = 2000
LEARNING_RATE = 0.0001
MAX_STEP = 500

def train():

    image_list, label_list = input_data.get_file(TRAIN_DIR)
    image_batch, label_batch = input_data.get_batch(image_list,
                                                    label_list,
                                                    IMG_H,IMG_W,
                                                    BATCH_SIZE, CAPACITY)
    logits = model_fcn.inference(image_batch, True)
    train_loss = model_fcn.loss(logits, label_batch)
    train_op = model_fcn.backpropagation(train_loss,  LEARNING_RATE)
    accuracy = model_fcn.accuracy(logits, label_batch)

    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(LOGS_DIR, sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()         # 多线程
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, loss, tra_acc = sess.run([train_op, train_loss, accuracy*100.0])


            if step % 50 == 0:
                print('step %d, train loss = %.2f, accuracy = %.2f%%' % (step, loss, tra_acc))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step+1) == MAX_STEP:
                checkpoint_path = os.path.join(LOGS_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


        coord.request_stop()

        coord.join(threads)

train()