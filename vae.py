import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from load_mnist import save_and_load_mnist
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import visdom
import shutil

#parameter setting
TOTAL_EPOCH = 30
BATCH_SIZE = 128
EPSILON = 1e-10

def inference(input_op, dim_z, reuse=False):
    with tf.variable_scope('inference', reuse=reuse):
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.zeros_initializer()

        in_W1 = tf.get_variable(name='in_W1', shape=[784, 500], initializer=w_init)
        in_b1 = tf.get_variable(name='in_b1', shape=[500], initializer=b_init)
        in_h1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(input_op, in_W1), in_b1), name='in_h1')

        in_W2 = tf.get_variable(name='in_W2', shape=[500, 500], initializer=w_init)
        in_b2 = tf.get_variable(name='in_b2', shape=[500], initializer=b_init)
        in_h2 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(in_h1, in_W2), in_b2), name='in_h2')

        in_W3 = tf.get_variable(name='in_W3', shape=[500, dim_z], initializer=w_init)
        in_b3 = tf.get_variable(name='in_b3', shape=[dim_z], initializer=b_init)
        z_mu = tf.nn.bias_add(tf.matmul(in_h2, in_W3), in_b3, name='z_mu')

        in_W4 = tf.get_variable(name='in_W4', shape=[500, dim_z], initializer=w_init)
        in_b4 = tf.get_variable(name='in_b4', shape=[dim_z], initializer=b_init)
        z_sigma = tf.nn.softplus(tf.nn.bias_add(tf.matmul(in_h2, in_W4), in_b4, name='z_sigma')) + EPSILON
    return z_mu, z_sigma


def generator(z, dim_z, reuse=False):
    with tf.variable_scope("generator", reuse=reuse):
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.zeros_initializer()

        g_W1 = tf.get_variable(name='g_W1', shape=[dim_z, 500], initializer=w_init)
        g_b1 = tf.get_variable(name='g_b1', shape=[500], initializer=b_init)
        g_h1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(z, g_W1), g_b1), name='g_h1')

        g_W2 = tf.get_variable(name='g_W2', shape=[500, 500], initializer=w_init)
        g_b2 = tf.get_variable(name='g_b2', shape=[500], initializer=b_init)
        g_h2 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(g_h1, g_W2), g_b2), name='g_h2')

        g_W3 = tf.get_variable(name='g_W3', shape=[500, 784], initializer=w_init)
        g_b3 = tf.get_variable(name='g_b3', shape=[784], initializer=b_init)
        y = tf.sigmoid(tf.nn.bias_add(tf.matmul(g_h2, g_W3), g_b3), name='y')
    return y


class Model(object):
    def __init__(self, sess):
        tf.set_random_seed(0)
        self.dim_z = 2
        self._build_net()
        self.sess = sess
        self.vis = visdom.Visdom()


    def _build_net(self):
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='X')
        self.z_in = tf.placeholder(tf.float32, shape=[None, self.dim_z], name='latent_variable')

        mu, sigma = inference(self.X, self.dim_z)
        self.z = mu+sigma*tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        self.y = generator(self.z, self.dim_z)
        self.output = tf.clip_by_value(self.y, EPSILON, 1-EPSILON)

        marginal_likelihood = tf.reduce_sum(self.X*tf.log(self.output)+(1-self.X)*tf.log(1-self.output), axis=1)
        KL_divergence = 0.5*tf.reduce_sum(1+tf.square(mu)+tf.square(sigma)-tf.log(tf.square(sigma)), axis=1)

        self.marginal_likelihood = tf.reduce_mean(marginal_likelihood)
        KL_divergence = tf.reduce_mean(KL_divergence)

        ELBO = self.marginal_likelihood - KL_divergence
        self.loss = -ELBO
        self.optim = tf.train.AdamOptimizer(0.001).minimize(self.loss)

        self.avg_loss = tf.placeholder(tf.float32)
        self.avg_loss_scalar = tf.summary.scalar('avg_loss', self.avg_loss)


    def fit(self, x_train, x_test):
        if not os.path.exists('./board4_1'):
            os.mkdir('./board4_1')
        shutil.rmtree('./board4_1')
        self.writer = tf.summary.FileWriter('./board4_1')
        self.writer.add_graph(self.sess.graph)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        total_step = int(len(x_train)/BATCH_SIZE)

        print(">>> Start Train ")
        for epoch in range(TOTAL_EPOCH):
            loss_per_epoch = 0
            marginal_per_epoch = 0

            np.random.seed(epoch)
            mask = np.random.permutation(len(x_train))
            for step in range(total_step):
                s = step * BATCH_SIZE
                t = (step + 1) * BATCH_SIZE
                c, l, _ = self.sess.run([self.loss, self.marginal_likelihood, self.optim], feed_dict={self.X: x_train[mask[s:t]]})
                loss_per_epoch += c / total_step
                marginal_per_epoch += l / total_step

            s = self.sess.run(self.avg_loss_scalar, feed_dict={self.avg_loss:loss_per_epoch})
            self.writer.add_summary(s, global_step=epoch)
            print("Epoch : [{:4d}/{:4d}], cost : {:.6f}, marginal_loss : {:.6f}".format(epoch + 1, TOTAL_EPOCH, loss_per_epoch, -marginal_per_epoch))

            x_pred = self.get_restruction(x_test[:225])
            real_img = plot_mnist(x_test[:225], 225)
            pred_img = plot_mnist(x_pred, 225)

            x = np.linspace(-1.5, 1.5, 15)
            y = np.linspace(-1.5, 1.5, 15)
            cnt = 0
            for i in x:
                for j in y:
                    tmp = np.array([[i, j]])
                    if cnt == 0:
                        tmps = tmp
                        cnt += 1
                    else:
                        tmps = np.append(tmps, tmp, axis=0)

            pred = self.predict(tmps)
            pred_im = plot_mnist(pred, 225, 3)

            if epoch == 0:
                self.vis.image(real_img, opts=dict(title='Real Image'))
            self.vis.image(pred_img, opts=dict(title='Epoch {} Reconstruction Image'.format(epoch + 1)))
            self.vis.image(pred_im, opts=dict(title='Epoch {} Gen Image'.format(epoch + 1)))


    def predict(self, sample_z):
        y = generator(self.z_in, self.dim_z, reuse=True)
        return self.sess.run(y, feed_dict={self.z_in: sample_z})


    def get_restruction(self, x_test):
        y = self.sess.run(self.y, feed_dict={self.X: x_test})
        return y


def plot_mnist(images, n_images, seed=0):
    images = np.reshape(images, [len(images), 28, 28])
    plt.gca().set_axis_off()
    h_num = int(np.sqrt(n_images))
    v_num = int(np.sqrt(n_images))
    v_list = []
    count = 0
    for j in range(v_num):
        h_list = []
        for i in range(h_num):
            h_list.append(images[count])
            count+=1
        tmp = np.hstack(h_list)
        v_list.append(tmp)
    im = np.vstack(v_list)
    return im


def main():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    dataset = save_and_load_mnist("./data/mnist/")

    x_train = dataset['train_data']
    x_test = dataset['test_data']
    m = Model(sess)
    m.fit(x_train, x_test)


if __name__ == "__main__":
    main()
