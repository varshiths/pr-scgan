import tensorflow as tf
import os


class BaseModel:
    def __init__(self, config):
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()

        self.build_model()
        self.init_saver()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess, model):
        print("Saving model {} ...".format(os.path.join(self.config.checkpoint_dir, model)))
        self.saver.save(sess, os.path.join(self.config.checkpoint_dir, model))
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess, model):
        print("Loading model {} ...".format(os.path.join(self.config.checkpoint_dir, model)))
        self.saver.restore(sess, os.path.join(self.config.checkpoint_dir, model))
        print("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self):
        # just copy the following line in your child class
        self.saver = tf.train.Saver()

    def build_model(self):
        raise NotImplementedError
