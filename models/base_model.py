import tensorflow as tf
import os
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import json

import pprint; ppr = pprint.PrettyPrinter()
sprint = lambda x: ppr.pprint(x)

class BaseModel:
    def __init__(self, config):

        print("Building model ...")

        self.config = config
        self.writer = tf.summary.FileWriter(self.config.log_dir)
        # init the global step
        self.init_global_step()

        self.build_model()

        self.init_saver()
        self.summary = tf.summary.merge_all()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess, model, verbose=False):

        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)

        model_path = os.path.join(self.config.checkpoint_dir, model)

        print("Saving model {}-{} ...".format(model_path, int(self.gs)))
        self.saver.save(sess, model_path, global_step=int(self.gs))
        with open(model_path + ".params", "w") as f:
            json.dump(self.config, f, indent=4)
        print("Model and params saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess, model, verbose=False):
        model_path = os.path.join(self.config.checkpoint_dir, model)

        print("Loading model {} ...".format(model_path))

        if  os.path.exists(model_path + ".meta") and \
            os.path.exists(model_path + ".index"):
            if verbose:
                print("---------------------------------------------------")
                print("Variables being restored: ")
                print("---------------------------------------------------")
                reader = tf.train.NewCheckpointReader(model_path)
                var_to_shape_map = reader.get_variable_to_shape_map()
                sprint(var_to_shape_map)
                # print_tensors_in_checkpoint_file(
                #     file_name=model_path, 
                #     tensor_name='', 
                #     all_tensors=True
                #     )
                print("---------------------------------------------------")
            # self.saver = tf.train.import_meta_graph( model_path + '.meta' )
            self.saver.restore(sess, model_path)
            print("Model loaded")
            self.gs = float(model_path[model_path.rfind("-")+1:])
            print("Restoring global step to %d" % self.gs)
        else:
            print("File not found ... Random initialization ...")
            sess.run(tf.global_variables_initializer())

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        self.gs = 0.0

    def igs(self, session=None, amount=1.0):
        self.gs += amount
        if self.gs % self.config.save_freq == 0 and session is not None and self.config.save is not None:
            self.save(session, self.config.save)

    def init_saver(self):
        # just copy the following line in your child class
        varlist = tf.trainable_variables()
        if len(varlist) != 0:
            self.saver = tf.train.Saver(varlist, max_to_keep=self.config.max_to_keep)

    def build_model(self):
        pass
