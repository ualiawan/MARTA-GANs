# -*- coding: utf-8 -*-
import tensorflow as tf

def generator(inputs, is_train=True, reuse=False):
    output_size = 256
    kernel = 4
    
    batch_size  = 64
    gf_dim = 16
    c_dim = 3
    weight_init  =  tf.random_normal_initializer(stddev=0.01)
    #gamma_init = tf.random_normal_initializer(1., 0.01)
    
    s2, s4, s8, s16, s32, s64 = int(output_size/2), int(output_size/4), int(output_size/8), int(output_size/16), int(output_size/32), int(output_size/64)

    
    with tf.variable_scope('generator', reuse=reuse):
        
        h0 = tf.layers.dense(inputs, units=gf_dim*32*s64*s64, activation=tf.identity, kernel_initializer=weight_init)
        h0 = tf.reshape(h0, [-1, s64, s64, gf_dim*32])
        h0 = tf.contrib.layers.batch_norm(h0, scale=True, is_training=is_train, scope="g_bn0")
        h0 = tf.nn.relu(h0)
        
        
        output1_shape = [batch_size, s32, s32, gf_dim*16]
        w_h1 = tf.get_variable('g_w_h1', [kernel, kernel, output1_shape[-1], int(h0.get_shape()[-1])], 
                              initializer=weight_init)
        b_h1 = tf.get_variable('g_b_h1', [output1_shape[-1]], initializer=tf.constant_initializer(0))
        h1 = tf.nn.conv2d_transpose(h0, w_h1, output_shape=output1_shape, strides=[1, 2, 2, 1],
                                         padding='SAME', name='g_h1_deconv2d') + b_h1
        h1 = tf.contrib.layers.batch_norm(h1, scale=True, is_training=is_train, scope="g_bn1")
        h1 = tf.nn.relu(h1)
        
        
        output2_shape = [batch_size, s16, s16, gf_dim*8]
        w_h2 = tf.get_variable('g_w_h2', [kernel, kernel, output2_shape[-1], int(h1.get_shape()[-1])], 
                              initializer=weight_init)
        b_h2 = tf.get_variable('g_b_h2', [output2_shape[-1]], initializer=tf.constant_initializer(0))
        h2 = tf.nn.conv2d_transpose(h1, w_h2, output_shape=output2_shape, strides=[1, 2, 2, 1],
                                         padding='SAME', name='g_h2_deconv2d') + b_h2
        h2 = tf.contrib.layers.batch_norm(h2, scale=True, is_training=is_train, scope="g_bn2")
        h2 = tf.nn.relu(h2)
        
        
        output3_shape = [batch_size, s8, s8, gf_dim*4]
        w_h3 = tf.get_variable('g_w_h3', [kernel, kernel, output3_shape[-1], int(h2.get_shape()[-1])], 
                              initializer=weight_init)
        b_h3 = tf.get_variable('g_b_h3', [output3_shape[-1]], initializer=tf.constant_initializer(0))
        h3 = tf.nn.conv2d_transpose(h2, w_h3, output_shape=output3_shape, strides=[1, 2, 2, 1],
                                         padding='SAME', name='g_h3_deconv2d') + b_h3
        h3 = tf.contrib.layers.batch_norm(h3, scale=True, is_training=is_train, scope="g_bn3")
        h3 = tf.nn.relu(h3)
        
        
        output4_shape = [batch_size, s4, s4, gf_dim*2]
        w_h4 = tf.get_variable('g_w_h4', [kernel, kernel, output4_shape[-1], int(h3.get_shape()[-1])], 
                              initializer=weight_init)
        b_h4 = tf.get_variable('g_b_h4', [output4_shape[-1]], initializer=tf.constant_initializer(0))
        h4 = tf.nn.conv2d_transpose(h3, w_h4, output_shape=output4_shape, strides=[1, 2, 2, 1],
                                         padding='SAME', name='g_h4_deconv2d') + b_h4
        h4 = tf.contrib.layers.batch_norm(h4, scale=True, is_training=is_train, scope="g_bn4")
        h4 = tf.nn.relu(h4)
        
        
        output5_shape = [batch_size, s2, s2, gf_dim*1]
        w_h5 = tf.get_variable('g_w_h5', [kernel, kernel, output5_shape[-1], int(h4.get_shape()[-1])], 
                              initializer=weight_init)
        b_h5 = tf.get_variable('g_b_h5', [output5_shape[-1]], initializer=tf.constant_initializer(0))
        h5 = tf.nn.conv2d_transpose(h4, w_h5, output_shape=output5_shape, strides=[1, 2, 2, 1],
                                         padding='SAME', name='g_h5_deconv2d') + b_h5
        h5 = tf.contrib.layers.batch_norm(h5, scale=True, is_training=is_train, scope="g_bn5")
        h5 = tf.nn.relu(h5)
        
        
        output6_shape = [batch_size, output_size, output_size, c_dim]
        w_h6 = tf.get_variable('g_w_h6', [kernel, kernel, output6_shape[-1], int(h5.get_shape()[-1])], 
                              initializer=weight_init)
        b_h6 = tf.get_variable('g_b_h6', [output6_shape[-1]], initializer=tf.constant_initializer(0))
        h6 = tf.nn.conv2d_transpose(h5, w_h6, output_shape=output6_shape, strides=[1, 2, 2, 1],
                                         padding='SAME', name='g_h6_deconv2d') + b_h6
        
        
        #logits = h6.outputs
        #h6.outputs = tf.nn.tanh(h6.outputs)
        
    return tf.nn.tanh(h6)

def discriminator(inputs, is_train=True, reuse=False):
    kernel = 5
    df_dim =16
    weight_init = tf.random_normal_initializer(stddev=0.01)
    #gamma_init = tf.random_normal_initializer(1., 0.01)
    alpha_lrelu = 0.2
    
    with tf.variable_scope('discriminator', reuse=reuse):
        w_h0 = tf.get_variable('d_w_h0', [kernel, kernel, 3,  df_dim], initializer=weight_init)
        b_h0 = tf.get_variable('d_b_h0', [df_dim], initializer=tf.constant_initializer(0))
        h0 = tf.nn.conv2d(inputs, w_h0, strides=[1,2,2,1], padding='SAME', name='d_h0_conv2d') + b_h0
        h0 = tf.nn.leaky_relu(h0, alpha_lrelu)
        
       
        w_h1 = tf.get_variable('d_w_h1', [kernel, kernel, h0.get_shape()[-1],  df_dim*2], initializer=weight_init)
        b_h1 = tf.get_variable('d_b_h1', [df_dim*2], initializer=tf.constant_initializer(0))
        h1 = tf.nn.conv2d(h0, w_h1, strides=[1,2,2,1], padding='SAME', name='d_h1_conv2d') + b_h1
        h1 = tf.contrib.layers.batch_norm(h1, is_training=is_train, scope="d_bn1")
        h1 = tf.nn.leaky_relu(h1, alpha_lrelu)
        
        
        w_h2 = tf.get_variable('d_w_h2', [kernel, kernel, h1.get_shape()[-1],  df_dim*4], initializer=weight_init)
        b_h2 = tf.get_variable('d_b_h2', [df_dim*4], initializer=tf.constant_initializer(0))
        h2 = tf.nn.conv2d(h1, w_h2, strides=[1,2,2,1], padding='SAME', name='d_h2_conv2d') + b_h2
        h2 = tf.contrib.layers.batch_norm(h2, is_training=is_train, scope="d_bn2")
        h2 = tf.nn.leaky_relu(h2, alpha_lrelu)
        
        
        w_h3 = tf.get_variable('d_w_h3', [kernel, kernel, h2.get_shape()[-1],  df_dim*8], initializer=weight_init)
        b_h3 = tf.get_variable('d_b_h3', [df_dim*8], initializer=tf.constant_initializer(0))
        h3 = tf.nn.conv2d(h2, w_h3, strides=[1,2,2,1], padding='SAME', name='d_h3_conv2d') + b_h3
        h3 = tf.contrib.layers.batch_norm(h3, is_training=is_train, scope="d_bn3")
        h3 = tf.nn.leaky_relu(h3, alpha_lrelu)
        
        
        global_max_h3 = tf.nn.max_pool(h3, [1,4,4,1], strides=[1,4,4,1], padding='SAME', name='d_h3_maxpool')
        global_max_h3 = tf.layers.flatten(global_max_h3, name='d_h3_flatten')
        
        
        w_h4 = tf.get_variable('d_w_h4', [kernel, kernel, h3.get_shape()[-1],  df_dim*16], initializer=weight_init)
        b_h4 = tf.get_variable('d_b_h4', [df_dim*16], initializer=tf.constant_initializer(0))
        h4 = tf.nn.conv2d(h3, w_h4, strides=[1,2,2,1], padding='SAME', name='d_h4_conv2d') + b_h4
        h4 = tf.contrib.layers.batch_norm(h4, is_training=is_train, scope="d_bn4")
        h4 = tf.nn.leaky_relu(h4, alpha_lrelu)
        
        
        global_max_h4 = tf.nn.max_pool(h4, [1,2,2,1], strides=[1,2,2,1], padding='SAME', name='d_h4_maxpool')
        global_max_h4 = tf.layers.flatten(global_max_h4, name='d_h4_flatten')
        
        
        w_h5 = tf.get_variable('d_w_h5', [kernel, kernel, h4.get_shape()[-1],  df_dim*32], initializer=weight_init)
        b_h5 = tf.get_variable('d_b_h5', [df_dim*32], initializer=tf.constant_initializer(0))
        h5 = tf.nn.conv2d(h4, w_h5, strides=[1,2,2,1], padding='SAME', name='d_h5_conv2d') + b_h5
        h5 = tf.contrib.layers.batch_norm(h5, is_training=is_train, scope="d_bn5")
        h5 = tf.nn.leaky_relu(h5, alpha_lrelu)
        
        
        global_max_h5 = tf.layers.flatten(h5, name='d_h5_flatten')
        
        features  = tf.concat([global_max_h3, global_max_h4, global_max_h5], -1, name='d_concat')
        h6 = tf.layers.dense(features, units=1, activation=tf.identity, kernel_initializer=weight_init, name='d_h6_dense')
                
        
        #logits =  h6.outputs
        #h6.outputs = tf.nn.sigmoid(h6.outputs)
    return tf.nn.sigmoid(h6), features

        
        
        
