import tensorflow as tf
from .base_model import BaseModel
import pprint

import pdb

pp = pprint.PrettyPrinter()


class SeqGAN(BaseModel):

    def create_placeholders(self):

        self.sequence = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.config.time_steps, self.config.sequence_width]
            )

    def rnn_unit_gen(self):

        def rnn_cell():
            return tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.BasicLSTMCell(num_units=self.config.lstm_units_gen),
            output_keep_prob=self.config.keep_prob,
            variational_recurrent=True,
            dtype=tf.float32
            )

        return tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(self.config.lstm_layers_gen)])

    def generator_network(self, state):

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):

            with tf.variable_scope("embedding_latent"):
                w = tf.get_variable("weight", [self.config.latent_state_size, self.config.embedding_latent])
                b = tf.get_variable("bias", [self.config.embedding_latent])

                embedding_latent = tf.nn.sigmoid( tf.matmul(state, w) + b )

            with tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
                rnn_unit_gen = self.rnn_unit_gen()

                with tf.variable_scope("input_embedding"):
                    wi = tf.get_variable("weight", [self.config.sequence_width + self.config.embedding_latent, self.config.lstm_input_gen])
                    bi = tf.get_variable("bias", [self.config.lstm_input_gen])

                with tf.variable_scope("output_embedding"):
                    wo = tf.get_variable("weight", [self.config.lstm_units_gen, self.config.sequence_width])
                    bo = tf.get_variable("bias", [self.config.sequence_width])

                outputs = []
                rnn_state = rnn_unit_gen.zero_state(self.config.batch_size, dtype=tf.float32)
                rnn_out = tf.zeros([self.config.batch_size, self.config.sequence_width])
                for _ in range(self.config.time_steps):
                    rnn_in = tf.concat([embedding_latent, rnn_out], axis=1)
                    rnn_in = tf.tanh(tf.matmul(rnn_in, wi) + bi)

                    rnn_out, rnn_state = rnn_unit_gen(rnn_in, rnn_state)
                    
                    rnn_out = tf.tanh(tf.matmul(rnn_out, wo) + bo)
                    outputs.append(rnn_out)

                outputs = tf.stack(outputs, axis=1)

        return outputs

    def discriminator_network(self, seq):

        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            pass

        return None

    def generator_cost(self, prob_image_gen):

        logt = tf.log( prob_image_gen + self.epsilon )
        cost = -tf.reduce_mean(logt)
        return cost

    def discriminator_cost(self, prob_image_gen, prob_image_target):

        logtg = tf.log( 1 - prob_image_gen + self.epsilon )
        logtt = tf.log( prob_image_target + self.epsilon )

        cost = -tf.reduce_mean( logtt + logtg )
        return cost

    def build_model(self):

        self.epsilon = tf.constant(1e-8)
        self.create_placeholders()

        latent_state = tf.random_uniform(
            [self.config.batch_size, self.config.latent_state_size],
            minval=0,
            maxval=1,
        )

        out_gen = self.out_gen = self.generator_network(latent_state)

        disc_out_gen = self.disc_out_gen = self.discriminator_network(out_gen)
        disc_out_target = self.disc_out_target = self.discriminator_network(self.sequence)

        # self.gen_cost = self.generator_cost(disc_out_gen)
        # self.disc_cost = self.discriminator_cost(disc_out_gen, disc_out_target)

        # self.build_gradient_steps()

        self.build_validation_metrics()

    def build_gradient_steps(self):

        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

        gen_grad_vars = gen_vars
        disc_grad_vars = disc_vars

        gen_grads = self.gen_grads = tf.gradients(self.gen_cost, gen_grad_vars)
        disc_grads = self.disc_grads = tf.gradients(self.disc_cost, disc_grad_vars)

        optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate
            )

        clipped_gen_grads, _ = tf.clip_by_global_norm(gen_grads, self.config.max_grad)
        clipped_disc_grads, _ = tf.clip_by_global_norm(disc_grads, self.config.max_grad)

        self.gen_grad_step = optimizer.apply_gradients(zip(clipped_gen_grads, gen_grad_vars))
        self.disc_grad_step = optimizer.apply_gradients(zip(clipped_disc_grads, disc_grad_vars))

    def build_validation_metrics(self):
        pass
