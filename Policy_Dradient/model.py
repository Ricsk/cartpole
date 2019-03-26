import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient(object):  # 首先创建一个模型类
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,  # 学习效率
                 reward_decay=0.9,  # 奖励衰减
                 replace_target_iter=5,  # 更新预测神经网络的周期
                 output_graph=False,  # 是否输出Tensorboard
                 layer1_elmts=20,  # 神经元个数
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.layer1_elmts = layer1_elmts
        self.learn_step_counter = 0

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # 这是我们存储 回合信息的 list

        self.build_net()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.cost_his = []  # 这里建立cost数组，图像显示cost

        if output_graph:
            tf.summary.FileWriter('./logs', self.sess.graph)

    def build_net(self):
        # 为什么不加偏置？
        self.state = tf.placeholder(tf.float32, [None, self.n_features], name="state")
        self.y_true = tf.placeholder(tf.float32, [None, ], name="y_true")
        self.advantages = tf.placeholder(tf.float32, [None, ], name="reward_signal")

        # w_initializer = tf.contrib.layers.xavier_initializer()
        w_initializer = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_initializer = tf.constant_initializer(0.1)
        with tf.variable_scope('net'):
            e_layer1 = tf.layers.dense(self.state, self.layer1_elmts, tf.nn.relu, kernel_initializer=w_initializer,
                                       bias_initializer=b_initializer, name='e_layer1')
            all_act = tf.layers.dense(e_layer1, self.n_actions, kernel_initializer=w_initializer,
                                       bias_initializer=b_initializer, name='e_layer2')
            self.y_prob = tf.nn.softmax(all_act, name='act_prob')

        with tf.variable_scope('loss'):
            # 最大化 总体 reward (log_p * R) 就是在最小化 -(log_p * R), 而 tf 的功能里只有最小化 loss
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act,
            #                                                               labels=self.y_true)  # 所选 action 的概率 -log 值
            # 下面的方式是一样的:
            neg_log_prob = tf.reduce_sum(-tf.log(self.y_prob)*tf.one_hot(tf.to_int32(self.y_true), self.n_actions), axis=1)
            self.loss = tf.reduce_mean(neg_log_prob * self.advantages)  # (vt = 本reward + 衰减的未来reward) 引导参数的梯度下降
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.y_prob, feed_dict={self.state: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def _discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        # reserved 返回的是列表的反序，这样就得到了贴现求和值。
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def learn(self):
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        self.sess.run(self.train_op, feed_dict={
            self.state: np.vstack(self.ep_obs),
            self.y_true: np.array(self.ep_as),
            self.advantages: discounted_ep_rs_norm,
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return discounted_ep_rs_norm

    def plot_cost(self):  # 显示Cost
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training step')
        plt.show()
