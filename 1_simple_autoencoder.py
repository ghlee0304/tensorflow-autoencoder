import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from load_mnist import save_and_load_mnist
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#parameter setting
TOTAL_EPOCH = 1
BATCH_SIZE = 32


class Model(object):
    def __init__(self, sess):
        tf.set_random_seed(0)
        self.build_net()
        self.sess = sess


    def build_net(self):
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='X')
        with tf.variable_scope("encoder"):
            W_encoder = tf.get_variable(name='W_encoder', shape=[784, 256], initializer=tf.glorot_uniform_initializer())
            b_encoder = tf.get_variable(name='b_encoder', shape=[256], initializer=tf.zeros_initializer())
            out_encoder = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.X, W_encoder), b_encoder), name='ont_encoder')

        with tf.variable_scope("decoder"):
            #W_decoder = tf.transpose(W_encoder, name='W_decoder')
            W_decoder = tf.get_variable(name='W_decoder', shape=[256, 784], initializer=tf.glorot_uniform_initializer())
            b_decoder = tf.get_variable(name='b_decoder', shape=[784], initializer=tf.zeros_initializer())
            self.out_decoder = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(out_encoder, W_decoder), b_decoder), name='out_decoder')

        self.optim = self.optimizer


    def fit(self, x_train):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        total_step = int(len(x_train)/BATCH_SIZE)

        print(">>> Start Train ")
        for epoch in range(TOTAL_EPOCH):
            loss_per_epoch = 0

            np.random.seed(epoch)
            mask = np.random.permutation(len(x_train))
            for step in range(total_step):
                s = step * BATCH_SIZE
                t = (step + 1) * BATCH_SIZE
                c, _ = self.sess.run([self.loss, self.optim], feed_dict={self.X: x_train[mask[s:t]]})
                loss_per_epoch += c / total_step

            print("Epoch : [{:4d}/{:4d}], cost : {:.6f}".format(epoch + 1, TOTAL_EPOCH, loss_per_epoch))


    def predict(self, x_test):
        return self.sess.run(self.out_decoder, feed_dict={self.X: x_test})


    @property
    def optimizer(self):
        return tf.train.AdamOptimizer(0.01).minimize(self.loss)


    @property
    def loss(self):
        return tf.reduce_mean(tf.square(self.X-self.out_decoder), name='loss')


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

    m = Model(sess)
    m.fit(x_train)

    x_pred = m.predict(x_test[:25])

    plot_mnist(x_test[:25], 25, 1)
    plot_mnist(x_pred, 25, 2)

    plt.show()


if __name__ == "__main__":
    main()
