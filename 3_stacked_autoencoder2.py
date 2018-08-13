from load_mnist import save_and_load_mnist
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

TOTAL_EPOCH = 5
PRE_EPOCH = 5
BATCH_SIZE = 32


class Stacked_autoencoder1(object):
    def __init__(self, sess):
        tf.set_random_seed(0)
        self._build_net()
        self.sess = sess


    def encoder(self, x, d, variable_name, name):
        with tf.variable_scope(name):
            input_shape = x.get_shape().as_list()
            W = tf.get_variable(name='{}_W'.format(variable_name), shape=[input_shape[-1], d], initializer=tf.glorot_uniform_initializer())
            W_t = tf.transpose(W)
            b = tf.get_variable(name='{}_b'.format(variable_name), shape=[d], initializer=tf.zeros_initializer())
            b_t = tf.get_variable(name='{}_b_t'.format(variable_name), shape=[input_shape[-1]], initializer=tf.zeros_initializer())
            h = tf.sigmoid(tf.nn.bias_add(tf.matmul(x, W), b), name='{}_h'.format(variable_name))
            o = tf.sigmoid(tf.nn.bias_add(tf.matmul(h, W_t), b_t, name='{}_o'.format(variable_name)))
        return h, o, W_t


    def decoder(self, x, W, variable_name, name):
        with tf.variable_scope(name):
            s = W.get_shape().as_list()
            b = tf.get_variable(name='{}_b'.format(variable_name), shape=[s[-1]], initializer=tf.zeros_initializer())
            h = tf.sigmoid(tf.nn.bias_add(tf.matmul(x, W), b, name='{}_h'.format(variable_name)))
        return h


    def _build_net(self):
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='X')
        self.learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        en1, en1_o, W1 = self.encoder(self.X, 256, 'en1', 'encoder1')
        en2, en2_o, W2 = self.encoder(en1, 128, 'en2', 'encoder2')
        de1 = self.decoder(en2, W2, 'de1')
        self.output = self.decoder(de1, W1, 'de2')

        self.en1_loss = self.loss(self.X, en1_o, 'en1_loss')
        self.en2_loss = self.loss(en1, en2_o, 'en2_loss')
        self.total_loss = self.loss(self.X, self.output, 'total_loss')

        t_vars = tf.trainable_variables()
        en1_vars = [var for var in t_vars if 'en1_' in var.name]
        en2_vars = [var for var in t_vars if 'en2_' in var.name]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.en1_optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.en1_loss, var_list=en1_vars)
            self.en2_optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.en2_loss, var_list=en2_vars)
            self.total_optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss, var_list=t_vars)


    def loss(self, x, y, name):
        with tf.variable_scope(name):
            l = tf.reduce_mean(tf.square(x-y), name='loss')
        return l


    def fit(self, x_train):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        total_step = int(len(x_train) / BATCH_SIZE)

        print("\n> Start Pretrain 1")
        for epoch in range(PRE_EPOCH):
            en1_loss_per_epoch = 0

            np.random.seed(epoch)
            mask = np.random.permutation(len(x_train))
            for step in range(total_step):
                s = step * BATCH_SIZE
                t = (step + 1) * BATCH_SIZE
                c1, _ = self.sess.run([self.en1_loss, self.en1_optim], feed_dict={self.X: x_train[mask[s:t]], self.learning_rate:0.01})
                en1_loss_per_epoch += c1 / total_step

            print("Loss1 : ", en1_loss_per_epoch)

        print("\n> Start Pretrain 2")
        for epoch in range(PRE_EPOCH):
            en2_loss_per_epoch = 0

            np.random.seed(epoch)
            mask = np.random.permutation(len(x_train))
            for step in range(total_step):
                s = step * BATCH_SIZE
                t = (step + 1) * BATCH_SIZE
                c2, _ = self.sess.run([self.en2_loss, self.en2_optim], feed_dict={self.X: x_train[mask[s:t]], self.learning_rate: 0.01})
                en2_loss_per_epoch += c2 / total_step

            print("Loss2 : ", en2_loss_per_epoch)

        print("Pretrain Done.\n")


        print("> Start Train ")
        for epoch in range(TOTAL_EPOCH):
            loss_per_epoch = 0

            np.random.seed(epoch)
            mask = np.random.permutation(len(x_train))
            for step in range(total_step):
                s = step * BATCH_SIZE
                t = (step + 1) * BATCH_SIZE
                c, _ = self.sess.run([self.total_loss, self.total_optim], feed_dict={self.X: x_train[mask[s:t]], self.learning_rate:0.001})
                loss_per_epoch += c / total_step

            print("Epoch : [{:4d}/{:4d}], cost : {:.6f}".format(epoch + 1, TOTAL_EPOCH, loss_per_epoch))


    def predict(self, x_test):
        return self.sess.run(self.output, feed_dict={self.X: x_test})


def plot_mnist(images, n_images, fig, seed=0):
    images = np.reshape(images, [len(images), 28, 28])
    plt.figure()
    plt.gca().set_axis_off()
    h_num = int(np.sqrt(n_images))
    v_num = int(np.sqrt(n_images))
    v_list = []
    np.random.seed(seed)
    mask = np.random.permutation(len(images))
    count = 0
    for j in range(v_num):
        h_list = []
        for i in range(h_num):
            h_list.append(images[mask[count]])
            count+=1
        tmp = np.hstack(h_list)
        v_list.append(tmp)
    im = np.vstack(v_list)
    plt.figure(fig)
    plt.imshow(im, cmap=plt.cm.gray_r, interpolation='nearest')


def main():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    dataset = save_and_load_mnist("./data/mnist/")

    x_train = dataset['train_data']
    x_test = dataset['test_data']

    m = Stacked_autoencoder1(sess)
    m.fit(x_train)

    x_pred = m.predict(x_test[:25])

    plot_mnist(x_test[:25], 25, 1)
    plot_mnist(x_pred, 25, 2)

    plt.show()


if __name__ == "__main__":
    main()
    
    
'''
> Start Pretrain 1
Loss1 :  0.009490525403215244
Loss1 :  0.004448691072408113
Loss1 :  0.004064533411214749
Loss1 :  0.0038575562359765203
Loss1 :  0.003707252243378506

> Start Pretrain 2
Loss2 :  0.0348969497556488
Loss2 :  0.02481857640420399
Loss2 :  0.023476903745283693
Loss2 :  0.02265851817776761
Loss2 :  0.022062871495882676
Pretrain Done.

> Start Train 
Epoch : [   1/   5], cost : 0.006497
Epoch : [   2/   5], cost : 0.005897
Epoch : [   3/   5], cost : 0.005585
Epoch : [   4/   5], cost : 0.005363
Epoch : [   5/   5], cost : 0.005187
'''
