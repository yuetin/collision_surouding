import numpy as np
import tensorflow as tf
import gym
import random

NAME = 'SAC_v15_9'
EPS = 1e-8

class ActorNetwork(object):
    def __init__(self, act_dim, name):
        self.act_dim = act_dim
        self.name = name

    def step(self, obs, depth, log_std_min=-5, log_std_max=0.5):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):


            Actor_cnn = FUCK_CNNNETWORK()
            img_cnn_input = depth
            img_input = Actor_cnn.fuck_cnn(img_cnn_input)
            fc_input = tf.concat([obs, img_input], axis=1)
            h1 = tf.layers.dense(fc_input, 1024, tf.nn.leaky_relu, name='h1')
            h2 = tf.layers.dense(h1, 512, tf.nn.leaky_relu, name='h2')
            h3 = tf.layers.dense(h2, 512, tf.nn.leaky_relu, name='h3')
            # h4 = tf.layers.dense(h3, 256, tf.nn.leaky_relu, name='h4')
            # h5 = tf.layers.dense(h4, 256, tf.nn.leaky_relu, name='h5')
            mu = tf.layers.dense(h3, self.act_dim, None, name='mu')
            log_std = tf.layers.dense(h3, self.act_dim, tf.tanh, name='log_std')
            log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

            std = tf.exp(log_std)
            pi = mu + tf.random_normal(tf.shape(mu)) * std

            mu = tf.tanh(mu)
            pi = tf.tanh(pi)

            # mu = tf.tanh(mu)
        return mu, pi

    def evaluate(self, obs, depth):
        mu, pi = self.step(obs, depth)
        return mu


class   FUCK_CNNNETWORK(object):
    def __init__(self):
        
        pass
        
        
        ## CNN 第一層
    def fuck_cnn(self, img_buffer):
        # arrayA = np.array(img_buffer)
        # print("aaaaaaaaaaa")
        # print(img_buffer)
        input_x_images=tf.reshape(img_buffer,[-1,320,240,1])
        input_x_images=tf.image.resize(input_x_images,(128,128),method=0)
        # input_x_images = img_buffer
        conv1=tf.layers.conv2d(
        inputs=input_x_images,
        filters=64,
        kernel_size=[5,5],
        strides=2,
        padding='same',
        trainable = 0,
        activation=tf.nn.relu
        )
        # print(conv1)
        # conv1=tf.layers.batch_normalization(conv1,training=True)

        ## 池化層 1  
        # pool1=tf.layers.max_pooling2d(
        # inputs=conv1,
        # pool_size=[2,2],
        # strides=2
        # )
        # print(pool1)

        ## CNN 第二層
        conv2=tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[5,5],
        strides=2,
        padding='same',
        trainable = 0,
        activation=tf.nn.relu
        )   
        # conv2=tf.layers.batch_normalization(conv2,training=True)

        # pool2=tf.layers.max_pooling2d(
        # inputs=conv2,
        # pool_size=[2,2],
        # strides=2
        # )

        ## CNN 第三層
        conv3=tf.layers.conv2d(
        inputs=conv2,
        filters=32,
        kernel_size=[3,3],
        strides=2,
        padding='same',
        trainable = 0,
        activation=tf.nn.relu
        )   
        # conv3=tf.layers.batch_normalization(conv3,training=True)

        # pool3=tf.layers.max_pooling2d(
        # inputs=conv3,
        # pool_size=[2,2],
        # strides=2
        # )

        ## CNN 第四層
        conv4= tf.layers.conv2d(
        inputs=conv3,
        filters=32,
        kernel_size=[3,3],
        strides=2,
        padding='same',
        trainable = 0,
        activation=tf.nn.relu
        )   
        # conv4=tf.layers.batch_normalization(conv4,training=True)

        # pool4=tf.layers.max_pooling2d(
        # inputs=conv4,
        # pool_size=[2,2],
        # strides=2
        # )
        
        # flat=tf.reshape(pool4,[-1,20*15*32])
        flat=tf.reshape(conv4,[-1,8*8*32])
        # flat=tf.reshape(pool3,[-1,4*3*32])
        dense_cnn=tf.layers.dense(
        inputs=flat,
        units=512,
        activation=tf.nn.leaky_relu
        )

        # dropout=tf.layers.dropout(
        # inputs=dense_cnn,
        # rate=0.5,
        # )

        # logits=tf.layers.dense(
        # inputs=dropout,
        # units=128
        # )

        # logits = tf.reshape(logits,[-1,128])
        # logits = tf.reshape(dropout,[-1,])
        return dense_cnn
        # dropout=tf.layers.dropout(
        # inputs=dense,
        # rate=0.5,
        # )


        # logits=tf.layers.dense(
        # inputs=dropout,
        # units=10
        # )       


        # return x

class SAC(object):
    def __init__(self, act_dim, obs_dim, depth_dim, name=None):
        # tf.reset_default_graph()

        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.name = name
        self.depth_dim = depth_dim

        self.OBS0 = tf.placeholder(tf.float32, [None, self.obs_dim], name=self.name+"observations0")
        self.DEPTH0 = tf.placeholder(tf.float32, [None, self.depth_dim], name =self.name+'depth_gif0')
        policy = ActorNetwork(self.act_dim, self.name+'Actor')

        self.mu = policy.evaluate(self.OBS0, self.DEPTH0)
        if self.name == 'right_':
            self.path = '/home/yue/yuetin/collision_surrouding/src/Collision_Avoidance/train/weights/'+ NAME +'/'+ self.name+'40'
        else:
            self.path = '/home/yue/yuetin/collision_surrouding/src/Collision_Avoidance/train/weights/'+ NAME +'/'+ self.name+'40'
        # self.path = '/home/yue/yuetin/collision_surrouding/src/Collision_Avoidance/train/weights/'+ NAME +'/'+ self.name+'40'

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.path))

    def choose_action(self, obs, depth):
        action = self.sess.run(self.mu, feed_dict={self.OBS0: obs.reshape(1, -1), self.DEPTH0: depth.reshape(1, -1)})
        action = np.squeeze(action)
        return action