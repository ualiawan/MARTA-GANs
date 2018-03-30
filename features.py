# -*- coding: utf-8 -*-
import tensorflow as tf
import network
import sys
import os
import numpy as np
from glob import glob
from random import shuffle
import utilities

flags = tf.app.flags

flags.DEFINE_integer("train_size", sys.maxsize, "The size of train images")
flags.DEFINE_integer("batch_size", 64, "The number of batch images")
flags.DEFINE_integer("image_size", 256, "The size of image to use (will be center cropped)")
flags.DEFINE_integer("output_size", 256, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("features_size", 14336, "Number of features for one image")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("sample_step", 500, "The interval of generating sample. [500]")
flags.DEFINE_string("train_dataset", "uc_train_256_data", "The name of dataset")
flags.DEFINE_string("test_dataset", "uc_test_256", "The name of dataset")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("feature_dir", "features", "Directory name to save features")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("num_labels", 21, "Number of different labels")
flags.DEFINE_string("labels_file", "style_names.txt", "File containing a list of labels")


FLAGS = flags.FLAGS


def main(_):
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        print("Houston tengo un problem: No checkPoint directory found")
        return 0
    if not os.path.exists(FLAGS.feature_dir):
        os.makedirs(FLAGS.feature_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
        
    #with tf.device("/gpu:0"):
    with tf.device("/cpu:0"):
        
        x =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim], name='d_input_images')
        
        d_netx, Dx, Dfx =  network.discriminator(x, is_train=FLAGS.is_train, reuse=False)
                
        saver = tf.train.Saver()
        
    with tf.Session() as sess:
        print("starting session")
        sess.run(tf.global_variables_initializer())
    
        model_dir = "%s_%s_%s" % (FLAGS.train_dataset, 64, FLAGS.output_size)
        save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
        labels =  utilities.get_labels(FLAGS.num_labels,FLAGS.labels_file)
        
        saver.restore(sess, save_dir)
        print("Model restored from file: %s" % save_dir)
        
        #extracting features from train dataset
        extract_features(x, labels, sess, Dfx)
        
        #extracting features from test dataset
        extract_features(x, labels, sess, Dfx, training=False)

    sess.close()


def extract_features(x, labels, sess, Dfx, training=True):
    if training:
        data_path = FLAGS.train_dataset
        features_path = "features_train.npy"
        labels_path = "labels_train.npy"
    else: 
        data_path = FLAGS.test_dataset
        features_path = "features_test.npy"
        labels_path = "labels_test.npy"
        
    data_files = glob(os.path.join("./data", data_path, "*.jpg"))
    shuffle(data_files)
    num_batches = min(len(data_files), FLAGS.train_size) // FLAGS.batch_size
    #num_batches =2
    
    num_examples = num_batches*FLAGS.batch_size
    y = np.zeros(num_examples, dtype=np.uint8)
    for i in range(num_examples):
        for j in range(len(labels)):
            if labels[j] in data_files[i]:
                y[i] = j
                break
    
    features = np.zeros((num_examples, FLAGS.features_size))
    
    for batch_i in range(num_batches):
        batch_files = data_files[batch_i*FLAGS.batch_size:(batch_i+1)*FLAGS.batch_size]
        batch = [utilities.load_image(batch_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size) for batch_file in batch_files]
        batch_x = np.array(batch).astype(np.float32)
        
        f = sess.run(Dfx, feed_dict={x: batch_x})
        begin = FLAGS.batch_size*batch_i
        end = FLAGS.batch_size + begin
        features[begin:end, ...] = f
    
    print("Features Extracted, Now saving")
    np.save(os.path.join(FLAGS.feature_dir, features_path), features)
    np.save(os.path.join(FLAGS.feature_dir, labels_path), y)
    
    print("Features Saved")
    
    sys.stdout.flush()
    
    
    
    
if __name__ == '__main__':
    tf.app.run()