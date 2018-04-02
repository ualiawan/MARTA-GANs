# -*- coding: utf-8 -*-

import tensorflow as tf
import network
import sys
import os
import numpy as np
from glob import glob
from random import shuffle
import utilities
import time


flags = tf.app.flags

flags.DEFINE_integer("epoch", 30, "Epoch to train")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_integer("train_size", sys.maxsize, "The size of train images")
flags.DEFINE_integer("batch_size", 64, "The number of batch images")
flags.DEFINE_integer("image_size", 256, "The size of image to use (will be center cropped)")
flags.DEFINE_integer("output_size", 256, "The size of the output images to produce")
flags.DEFINE_integer("sample_size", 64, "The number of sample images")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color")
flags.DEFINE_integer("z_dim", 100, "Dimensions of input niose to generator")
flags.DEFINE_integer("sample_step", 500, "The interval of generating sample")
flags.DEFINE_string("dataset", "uc_train_256_data", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("summaries_dir", "logs", "Directory name to save the summaries")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

def main(_):
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.summaries_dir):
        os.makedirs(FLAGS.summaries_dir)
        
    with tf.device("/gpu:0"):
    #with tf.device("/cpu:0"):
        z = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.z_dim], name="g_input_noise")
        x =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim], name='d_input_images')
        
        Gz =  network.generator(z)
        Dx, Dfx =  network.discriminator(x)
        Dz, Dfz = network.discriminator(Gz, reuse=True)
        
        
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dz, labels=tf.zeros_like(Dz)))
        d_loss =  d_loss_real + d_loss_fake
        
        g_loss_perceptual = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dz, labels = tf.ones_like(Dz)))
        g_loss_features = tf.reduce_mean(tf.nn.l2_loss(Dfx-Dfz))/(FLAGS.image_size*FLAGS.image_size)
        g_loss = g_loss_perceptual + g_loss_features
        
        
        tvars = tf.trainable_variables()
        d_vars =  [var for var in tvars if 'd_' in var.name]
        g_vars =  [var for var in tvars if 'g_' in var.name]

        print(d_vars)
        print("---------------")
        print(g_vars)
        
        with tf.variable_scope(tf.get_variable_scope(),reuse=False): 
            print("reuse or not: {}".format(tf.get_variable_scope().reuse))
            assert tf.get_variable_scope().reuse == False, "Houston tengo un problem"
            d_trainer = tf.train.AdamOptimizer(FLAGS.learning_rate, FLAGS.beta1).minimize(d_loss, var_list=d_vars)
            g_trainer = tf.train.AdamOptimizer(FLAGS.learning_rate, FLAGS.beta1).minimize(g_loss, var_list=g_vars)
        
        tf.summary.scalar("generator_loss_percptual", g_loss_perceptual)
        tf.summary.scalar("generator_loss_features", g_loss_features)
        tf.summary.scalar("generator_loss_total", g_loss)
        tf.summary.scalar("discriminator_loss", d_loss)
        tf.summary.scalar("discriminator_loss_real", d_loss_real)
        tf.summary.scalar("discriminator_loss_fake", d_loss_fake)
        
        images_for_tensorboard = network.generator(z, reuse=True)
        tf.summary.image('Generated_images', images_for_tensorboard, 2)
        
        merged = tf.summary.merge_all()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.30)
        gpu_options.allow_growth = True
              
        saver = tf.train.Saver()
        
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        
        print("starting session")
        summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
        sess.run(tf.global_variables_initializer())
        
        
        data_files = glob(os.path.join("./data", FLAGS.dataset, "*.jpg"))
        
        model_dir = "%s_%s_%s" % (FLAGS.dataset, 64, FLAGS.output_size)
        save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
        
        
        if FLAGS.is_train:
            for epoch in range(FLAGS.epoch):
                
                d_total_cost = 0.
                g_total_cost = 0.
                shuffle(data_files)
                num_batches = min(len(data_files), FLAGS.train_size) // FLAGS.batch_size
                #num_batches = 2
                for batch_i in range(num_batches):
                    batch_files = data_files[batch_i*FLAGS.batch_size:(batch_i+1)*FLAGS.batch_size]
                    batch = [utilities.load_image(batch_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size) for batch_file in batch_files]
                    batch_x = np.array(batch).astype(np.float32)
                    batch_z = np.random.normal(-1, 1, size=[FLAGS.batch_size, FLAGS.z_dim]).astype(np.float32)
                    start_time = time.time()
                    
                    d_err, _ = sess.run([d_loss, d_trainer], feed_dict={z: batch_z, x: batch_x})
                    g_err, _ = sess.run([g_loss, g_trainer], feed_dict={z: batch_z, x: batch_x})
                    
                    d_total_cost += d_err
                    g_total_cost += g_err
                    
                    if batch_i % 10 == 0:
                        summary = sess.run(merged, feed_dict={x: batch_x, z: batch_z})
                        summary_writer.add_summary(summary, (epoch-1)*(num_batches/30)+(batch_i/30))
                    
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                        % (epoch, FLAGS.epoch, batch_i, num_batches,
                            time.time() - start_time, d_err, g_err))
                

                print("Epoch:", '%04d' % (epoch+1), "d_cost=", \
                          "{:.9f}".format(d_total_cost/num_batches), "g_cost=", "{:.9f}".format(g_total_cost/num_batches))
    
                sys.stdout.flush()
        save_path = saver.save(sess, save_dir)
        print("Model saved in path: %s" % save_path)
        sys.stdout.flush()
    sess.close()   

if __name__ == '__main__':
    tf.app.run()
