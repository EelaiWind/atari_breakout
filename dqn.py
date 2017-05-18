from __future__ import print_function

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

    ''' You may need 3 placeholders for input: 4 input images, target Q value, action index
    def _build_network(self):
        # Placeholders for our input
        # Our input are 4 grayscale frames of shape 84, 84 each
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
    '''

def update_target_network(sess, behavior_Q, target_Q):
    # copy weights from behavior Q-network to target Q-network
    # Hint:
    #   * tf.trainable_variables()                  https://www.tensorflow.org/api_docs/python/tf/trainable_variables
    #   * variable.name.startswith(scope_name)      https://docs.python.org/3/library/stdtypes.html#str.startswith
    #   * assign                                    https://www.tensorflow.org/api_docs/python/tf/assign
    pass
