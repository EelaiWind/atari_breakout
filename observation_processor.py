from __future__ import print_function

import tensorflow as tf

class ObservationProcessor():
    """
    Processes a raw Atari image. Resizes it and converts it to grayscale.
    """
    def __init__(self):
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)              # input image
            self.output = tf.image.rgb_to_grayscale(self.input_state)                           # rgb to grayscale
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)           # crop image
            self.output = tf.image.resize_images(                                               # resize image
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            self.output = tf.squeeze(self.output)                                               # remove rgb dimension

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [84, 84, 1] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })
