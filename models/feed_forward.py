import tensorflow as tf
from .base_model import BaseModel
import pprint

pp = pprint.PrettyPrinter()


class FF(BaseModel):

    def create_placeholders(self):

        self.label = tf.placeholder(
                dtype=tf.float32,
                shape=[None, 10]
            )

        self.image = tf.placeholder(
                dtype=tf.float32,
                shape=[None, 784]
            )

    def create_embedding(self, inp):

        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            w = tf.get_variable("weight", [self.config.input_size, self.config.output_size])
            b = tf.get_variable("bias", [self.config.output_size])

            embed = tf.nn.softmax(tf.matmul( inp, w ) + b)

        return embed


    def build_model(self):

        self.epsilon = tf.constant(1e-8)
        self.create_placeholders()

        out = self.out = self.create_embedding(self.image)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.out))

        self.build_gradient_steps()

        self.build_validation_metrics()

    def build_gradient_steps(self):

        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        grads = tf.gradients(self.cost, tvars)

        optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.config.learning_rate
            )

        clipped_grads, _ = tf.clip_by_global_norm(grads, self.config.max_grad)

        self.train_step = optimizer.apply_gradients(zip(clipped_grads, tvars))

    def build_validation_metrics(self):

        eq_bools = tf.equal(tf.argmax(self.out,1), tf.argmax(self.label,1))
        self.validation_error = 100.0 * tf.reduce_mean( tf.cast(eq_bools, dtype=tf.float32) )

