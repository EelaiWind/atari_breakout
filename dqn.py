from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

class DQN():
    # Define the following things about Deep Q Network here:
    #   1. Network Structure (Check lab spec for details)
    #       * tf.contrib.layers.conv2d()
    #       * tf.contrib.layers.flatten()
    #       * tf.contrib.layers.fully_connected()
    #       * You may need to use tf.variable_scope in order to set different variable names for 2 Q-networks
    #   2. Target value & loss
    #   3. Network optimizer: tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
    #   4. Training operation for tensorflow

    def __init__(self, network_name, action_space, save_directory):
        self.m_parameter_list = []
        self.m_state_feature = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="state_feature")
        self.m_target_value = tf.placeholder(shape=[None], dtype=tf.float32, name="truth")
        self.m_selected_action = tf.placeholder(shape=[None], dtype=tf.int32, name="selected_action")

        self.m_optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.m_action_space = action_space
        with tf.variable_scope(network_name):
            self._build_network()

        if save_directory is not None:
            self.m_saver = tf.train.Saver(self.get_all_parameters())
            print("[INFO] Saving followning variables in checkpoint")
            for var in self.m_saver._var_list:
                print("  %s" % var.op.name)
            self.m_save_path = os.path.join(save_directory,'model')
            if not os.path.exists(save_directory):
                print("[INFO] Create \"%s\" for saving checkpoint" % save_directory)
                os.makedirs(save_directory)
            else:
                print("[WARNING] Checkpoint saving path \"%s\" already exists" % save_directory)

    def _make_variable(self, tensor_shape):
        weight = tf.get_variable(
            name="weight",
            shape=tensor_shape,
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(0.0, 0.01)
        )
        bias = tf.get_variable(
            name="bias",
            shape=tensor_shape[-1:],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(0.0, 0.01)
            #initializer=tf.constant_initializer(0.0)
        )
        self.m_parameter_list.append(weight)
        self.m_parameter_list.append(bias)
        return weight, bias

    def _get_loss(self):
        index = tf.transpose(tf.stack([tf.range(tf.shape(self.m_selected_action)[0]), self.m_selected_action], axis=0))
        q_value = tf.gather_nd(self.m_q_value, index)
        difference = tf.squared_difference(q_value, self.m_target_value)
        return tf.reduce_mean(difference)

    def _build_network(self):
        if hasattr(self, "m_update"): return

        input_tensor = tf.to_float(self.m_state_feature)/255.0
        with tf.variable_scope("conv_1"):
            weight, bias = self._make_variable([8, 8, 4, 32])
            conv_1 = tf.nn.conv2d(input_tensor, filter=weight, strides=[1,4,4,1], padding="SAME")
            conv_1 = tf.nn.bias_add(conv_1, bias)
            conv_1 = tf.nn.relu(conv_1)
        with tf.variable_scope("conv_2"):
            weight, bias = self._make_variable([4, 4, 32, 64])
            conv_2 = tf.nn.conv2d(conv_1, filter=weight, strides=[1,2,2,1], padding="SAME")
            conv_2 = tf.nn.bias_add(conv_2, bias)
            conv_2 = tf.nn.relu(conv_2)
        with tf.variable_scope("conv_3"):
            weight, bias = self._make_variable([3, 3, 64, 64])
            conv_3 = tf.nn.conv2d(conv_2, filter=weight, strides=[1,1,1,1], padding="SAME")
            conv_3 = tf.nn.bias_add(conv_3, bias)
            conv_3 = tf.nn.relu(conv_3)
        with tf.variable_scope("flatten"):
            shape = conv_3.get_shape().as_list()
            dimension = np.prod(shape[1:])
            flatten = tf.reshape(conv_3, [-1, dimension])
        with tf.variable_scope("fc_4"):
            weight, bias = self._make_variable([dimension, 512])
            fc_4 = tf.matmul(flatten, weight)
            fc_4 = tf.nn.bias_add(fc_4, bias)
            fc_4 = tf.nn.relu(fc_4)
        with tf.variable_scope("fc_5"):
            weight, bias = self._make_variable([512, self.m_action_space])
            fc_5 = tf.matmul(fc_4, weight)
            fc_5 = tf.nn.bias_add(fc_5, bias)

        self.m_q_value = fc_5
        self.m_loss = self._get_loss()
        self.m_update = self.m_optimizer.minimize(self.m_loss)

    def forward(self, session, state_feature):
        return session.run(self.m_q_value, feed_dict={self.m_state_feature:state_feature})

    def update(self, session, state_feature, selected_action, target_value):
        feed_dict = {
            self.m_state_feature:state_feature,
            self.m_selected_action: selected_action,
            self.m_target_value: target_value,
        }
        loss, _ = session.run([self.m_loss, self.m_update], feed_dict)
        return loss

    def get_all_parameters(self):
        return self.m_parameter_list

    def copy_parameter_from(self, session, source_network):
        # copy weights from behavior Q-network to target Q-network
        # Hint:
        #   * tf.trainable_variables()                  https://www.tensorflow.org/api_docs/python/tf/trainable_variables
        #   * variable.name.startswith(scope_name)      https://docs.python.org/3/library/stdtypes.html#str.startswith
        #   * assign                                    https://www.tensorflow.org/api_docs/python/tf/assign
        assign_operation = []
        for source, destination in zip(source_network.get_all_parameters(), self.get_all_parameters()):
            assign_operation.append(tf.assign(destination, source))
        session.run([assign_operation])

    def save_model(self, session, global_step):
        self.m_saver.save(session, self.m_save_path, global_step=global_step)

    def load_model(self, session, checkpoint_path):
        self.m_saver.restore(session, checkpoint_path)
