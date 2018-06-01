import tensorflow as tf
import os
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import json


class BaseModel:
    def __init__(self, config):

        print("Building model ...")

        self.config = config
        # init the global step
        self.init_global_step()

        self.build_model()

        self.init_saver()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess, model, verbose=False):

        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)

        model_path = os.path.join(self.config.checkpoint_dir, model)

        print("Saving model {} ...".format(model_path))
        self.saver.save(sess, model_path)
        print("Model saved")
        with open(model_path + ".params", "w") as f:
            json.dump(self.config, f)
        print("Params saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess, model, verbose=False):
        model_path = os.path.join(self.config.checkpoint_dir, model)

        print("Loading model {} ...".format(model_path))
        if verbose:
            print("---------------------------------------------------")
            print("Variables being restored: ")
            print("---------------------------------------------------")
            print_tensors_in_checkpoint_file(
                file_name=model_path, 
                tensor_name='', 
                all_tensors=True
                )
            print("---------------------------------------------------")

        try:
            # self.saver = tf.train.import_meta_graph( model_path + '.meta' )
            self.saver.restore(sess, model_path)
            print("Model loaded")
        except Exception as e:
            print("File not found ... Random initialization ...")
            sess.run(tf.global_variables_initializer())

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self):
        # just copy the following line in your child class
        self.saver = tf.train.Saver()

    def build_model(self):
        pass
