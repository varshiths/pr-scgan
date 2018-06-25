
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cProfile
import pstats
from tensorflow.python.client import device_lib

import tensorflow as tf
from models import *
from utils.config import process_config
from data import *
from data.utils import *

from utils.tests import *
from utils.tests2 import *
from utils.tests3 import *

import numpy as np


flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("seed", None, "seed to ensure reproducibility")
flags.DEFINE_string("architecture", None, "architecture to use")
flags.DEFINE_boolean("log", False, "log data and summaries for use with tensorboard")

flags.DEFINE_string("config", None, "config file for hyper-parameters")
flags.DEFINE_string("dataset", None, "dataset used for the model")
flags.DEFINE_string("mode", "dummy", "flag whether to train or test")
flags.DEFINE_string("model", None, "loading saved model")
flags.DEFINE_string("save", None, "name of the model to save")

flags.DEFINE_integer("test_index", None, "to select which test to perform")

FLAGS = flags.FLAGS


def main(argv):

    # build config
    config = process_config(FLAGS.config); config.train_phase = "train" == FLAGS.mode; config.log = FLAGS.log; config.save = FLAGS.save
    config.ngpus = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        )
    session_config.gpu_options.allow_growth = True

    if FLAGS.dataset == "mnist":
        data = MNIST(config)
    elif FLAGS.dataset == "jsl":
        data = JSL(config)
    elif FLAGS.dataset == "jslw":
        data = JSLW(config)
    elif FLAGS.dataset == "jsla":
        data = JSLA(config)

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        if FLAGS.seed is not None:
            tf.set_random_seed(FLAGS.seed)
            np.random.seed(FLAGS.seed)

        # create model
        if FLAGS.architecture == "ff":
            model = FF(config)
        elif FLAGS.architecture == "gan":
            model = GAN(config)
        elif FLAGS.architecture == "sgan":
            model = SGAN(config)
        elif FLAGS.architecture == "sganconv":
            model = SGANConv(config)
        elif FLAGS.architecture == "seqgan":
            model = SeqGAN(config)
        elif FLAGS.architecture == "cseqgan":
            model = CSeqGAN(config)
        else:
            model = BaseModel(config)

        with tf.Session(config=session_config) as session:

            session.run(tf.global_variables_initializer())
            if FLAGS.model is not None:
                model.load(session, FLAGS.model, verbose=False)

            if FLAGS.log:
                model.writer.add_graph(session.graph)
                model.writer.add_session_log(
                    tf.SessionLog(status=tf.SessionLog.START),
                    global_step=model.gs)

            if FLAGS.mode == "train":
                if FLAGS.architecture == "ff":
                    model_train = FFTrain(session, model, data, config, None)
                elif FLAGS.architecture == "gan":
                    model_train = GANTrain(session, model, data, config, None)
                elif FLAGS.architecture == "seqgan":
                    model_train = SeqGANTrain(session, model, data, config, None)
                elif FLAGS.architecture[:4] == "sgan":
                    model_train = SGANTrain(session, model, data, config, None)
                elif FLAGS.architecture == "cseqgan":
                    model_train = CSeqGANTrain(session, model, data, config, None)
                
                try:
                    model_train.train()
                except Exception as e:
                    raise e
                except KeyboardInterrupt as e:
                    print("Interrupted")
                    pass

                if FLAGS.save is not None:
                    model.save(session, FLAGS.save)
            elif FLAGS.mode[:4] == "test":
                # MNIST GAN Samples, Linear and SLERP Interpolation
                if FLAGS.test_index == 0:
                    run_model_and_plot_image(session, model, data, config)
                # JSL GAN Samples
                elif FLAGS.test_index == 1:
                    run_model_and_plot_gesture(session, model, data, config)
                # Output GAN Samples as CSV
                elif FLAGS.test_index == 2:
                    run_model_and_output_csv(session, model, data, config, "output/wgan")
                elif FLAGS.test_index == 3:
                    run_quart_model_and_plot_gesture(session, model, data, config)
                elif FLAGS.test_index == 4:
                    run_quart_model_and_output_csv(session, model, data, config, "output/csv_qwgan")
                elif FLAGS.test_index == 5:
                    run_data_norm_and_denorm(session, model, data, config)
                elif FLAGS.test_index == 6:
                    run_cseqgan_and_output_org_out(session, model, data, config, "output/csv_cseq")
                elif FLAGS.test_index == -1:
                    etc(session, model, data, config)

if __name__ == '__main__':
    try:
        cProfile.run("tf.app.run()", "/tmp/profdump")
    except Exception as e:
        import traceback
        traceback.print_exc()

    with open("run.prof", "w") as f:
        p = pstats.Stats("/tmp/profdump", stream=f)
        p.strip_dirs()
        p.sort_stats("cumulative")
        p.print_stats()
