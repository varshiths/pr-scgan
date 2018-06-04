import tensorflow as tf
import numpy as np
from .base_model import BaseModel
import pprint

pp = pprint.PrettyPrinter()


class SeqGAN(BaseModel):

    def create_placeholders(self):
        # sequence
        self.data = tf.placeholder(
                dtype=tf.float32,
                shape=[self.config.batch_size, self.config.time_steps, self.config.sequence_width]
            )

        self.latent = tf.placeholder(
                dtype=tf.float32,
                shape=[self.config.batch_size, self.config.latent_state_size]
            )

        self.start = tf.placeholder(
                dtype=tf.float32,
                shape=[self.config.batch_size, self.config.sequence_width]
            )

    def cnn_unit(params, activation=None, padding="SAME"):

        def get_variable(filt_dim, scope):
            return tf.get_variable(scope + "/filter", filt_dim)

        def function(inp):
            out = inp
            for i, (filt_dim, strides, dilations) in enumerate(params):
                out = tf.nn.conv2d(
                    out, 
                    get_variable(filt_dim, "conv%d" % (i)), 
                    data_format="NCHW",
                    strides=strides,
                    dilations=dilations,
                    padding=padding
                    )
                if activation is not None:
                    out = tf.nn.leaky_relu(out)
            return out

        return function

    def rnn_unit(num_units, num_layers, keep_prob):

        def rnn_cell():
            return tf.contrib.rnn.DropoutWrapper(
            tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=num_units),
            output_keep_prob=keep_prob,
            variational_recurrent=True,
            dtype=tf.float32
            )

        return tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(num_layers)])

    def generator_network(self, state, start, define=False):

        with tf.variable_scope("generator", reuse=(not define)):
            batch_size = state.shape[0].value

            with tf.variable_scope("embedding_latent"):
                w = tf.get_variable("weight", [self.config.latent_state_size, self.config.embedding_latent])
                b = tf.get_variable("bias", [self.config.embedding_latent])

                embedding_latent = tf.nn.leaky_relu( tf.matmul(state, w) + b )

            with tf.variable_scope("rnn", reuse=None) as rnn_scope:
                cell = SeqGAN.rnn_unit(
                    self.config.lstm_units_gen, 
                    self.config.lstm_layers_gen,
                    self.config.keep_prob
                    )

                with tf.variable_scope("input_embedding"):
                    wi = tf.get_variable("weight", [self.config.embedding_latent+self.config.sequence_width, self.config.lstm_input_gen])
                    bi = tf.get_variable("bias", [self.config.lstm_input_gen])

                '''
                vastly simplified model
                to be improved
                '''
                # input_seqs = tf.nn.leaky_relu(tf.matmul(embedding_latent, wi) + bi)
                # input_seqs = tf.reshape(input_seqs, [-1, 1, self.config.lstm_input_gen])
                # input_seqs = input_seqs + \
                #     np.zeros(
                #         shape=(batch_size, self.config.time_steps, self.config.lstm_input_gen), 
                #         )

                # outputs, _ = tf.nn.dynamic_rnn(
                #     cell, 
                #     input_seqs,
                #     initial_state=cell.zero_state(batch_size, dtype=tf.float32),
                #     dtype=tf.float32,
                #     )

                # outputs = tf.reshape(outputs, [-1, self.config.lstm_input_gen])
                # outputs = tf.nn.tanh( tf.matmul(outputs, wo) + bo )
                # outputs = tf.reshape(outputs, [-1, self.config.time_steps, self.config.sequence_width])

                '''
                Not compatible with tf 1.4.1
                '''
                def initialize():
                    next_inputs = tf.concat([embedding_latent, start], axis=1)
                    next_inputs = tf.nn.leaky_relu(tf.matmul(next_inputs, wi) + bi)
                    finished = tf.tile([False], [batch_size])
                    return (finished, next_inputs)

                def sample(time, outputs, state):
                    samples = tf.tile([0], [batch_size])
                    return samples

                def next_inputs(time, outputs, state, sample_ids):
                    
                    next_inputs = tf.concat([embedding_latent, outputs], axis=1)
                    next_inputs = tf.nn.leaky_relu(tf.matmul(next_inputs, wi) + bi)
                    finished = tf.tile([False], [batch_size])

                    # return (finished, next_inputs, next_state)
                    return (finished, next_inputs, state)

                helper = tf.contrib.seq2seq.CustomHelper(
                        initialize_fn=initialize,
                        sample_fn=sample,
                        next_inputs_fn=next_inputs,
                        sample_ids_shape=None,
                        sample_ids_dtype=None
                    )

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=cell,
                    helper=helper,
                    initial_state=cell.zero_state(batch_size, tf.float32),
                    output_layer=tf.layers.Dense(
                            units=self.config.sequence_width,
                            activation=tf.nn.leaky_relu,
                            kernel_initializer=tf.random_normal_initializer(),
                            bias_initializer=tf.random_normal_initializer(),
                            name="output_embedding",
                        ),
                    )
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder,
                    output_time_major=False,
                    parallel_iterations=12,
                    maximum_iterations=self.config.time_steps)

                outputs = outputs.rnn_output

                '''
                Unbelievably slow compile time
                Do not use
                '''
                # outputs = []
                # rnn_state = cell.zero_state(batch_size, dtype=tf.float32)
                # rnn_out = self.start
                # for i in range(self.config.time_steps):
                #     # reuse set
                #     if i > 0:
                #         rnn_scope.reuse_variables()

                #     rnn_in = tf.concat([embedding_latent, rnn_out], axis=1)
                #     rnn_in = tf.nn.leaky_relu(tf.matmul(rnn_in, wi) + bi)

                #     rnn_out, rnn_state = cell(rnn_in, rnn_state)
                    
                #     rnn_out = tf.nn.tanh(tf.matmul(rnn_out, wo) + bo)
                #     outputs.append(rnn_out)

                # outputs = tf.stack(outputs, axis=1)

        return outputs

    def discriminator_network(self, seq, define=False):

        with tf.variable_scope("discriminator", reuse=(not define)):

            layers_params = [
                ([5, 5, 1, 3], [1, 1, 1, 1], [1, 1, 1, 1]),
                ([5, 5, 3, 5], [1, 1, 1, 1], [1, 1, 1, 1]),
                ([5, 5, 5, 3], [1, 1, 1, 1], [1, 1, 1, 1]),
                ([5, 5, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1])
            ]

            cnn_unit_disc = SeqGAN.cnn_unit(layers_params, "leaky_relu", "SAME")

            seq = tf.reshape(seq, [-1, 1, self.config.time_steps, self.config.sequence_width])
            out_cnn = cnn_unit_disc(seq)

            # skip connection
            out_cnn += seq
            
            out_pool = tf.reshape(
                tf.nn.avg_pool(
                    out_cnn, 
                    [1, 1, out_cnn.shape[1].value, 1],
                    strides=[1,1,1,1],
                    padding="VALID",
                    ), 
                [-1, self.config.sequence_width]
                )
            
            with tf.variable_scope("prob_embed"):
                w = tf.get_variable("weight", [self.config.sequence_width, 1])
                b = tf.get_variable("bias", [1])

                out = tf.nn.sigmoid( tf.matmul(out_pool, w) + b )
                out = tf.reshape(out, [-1])

        return out

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

        out_gen = self.out_gen = self.generator_network(self.latent, self.start, True)

        disc_out_gen = self.disc_out_gen = self.discriminator_network(out_gen, True)
        disc_out_target = self.disc_out_target = self.discriminator_network(self.data)

        self.gen_cost = self.generator_cost(disc_out_gen)
        self.disc_cost = self.discriminator_cost(disc_out_gen, disc_out_target)

        self.build_gradient_steps()

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
