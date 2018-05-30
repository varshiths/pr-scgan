import tensorflow as tf
from .base_model import BaseModel
import pprint

pp = pprint.PrettyPrinter()


class SeqGAN(BaseModel):

    def create_placeholders(self):
        # sequence
        self.data = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.config.time_steps, self.config.sequence_width]
            )

        self.latent = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.config.latent_state_size]
            )

    def cnn_unit(params, activation=None, padding="SAME"):

        def get_variable(filt_dim, scope):
            return tf.get_variable(scope + "/filter", filt_dim)

        def function(inp):
            out = inp
            for i, (filt_dim, dilations) in enumerate(params):
                out = tf.nn.conv2d(
                    out, 
                    get_variable(filt_dim, "conv%d" % (i)), 
                    strides=[1,1,1,1], 
                    padding=padding
                    )
                if activation is not None:
                    out = leaky_relu(out)
            return out

        return function

    def rnn_unit(num_units, num_layers, keep_prob):

        def rnn_cell():
            return tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.BasicLSTMCell(num_units=num_units),
            output_keep_prob=keep_prob,
            # variational_recurrent=True,
            # dtype=tf.float32
            )

        return tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(num_layers)])

    def generator_network(self, state, define=False):

        with tf.variable_scope("generator", reuse=(not define)):

            with tf.variable_scope("embedding_latent"):
                w = tf.get_variable("weight", [self.config.latent_state_size, self.config.embedding_latent])
                b = tf.get_variable("bias", [self.config.embedding_latent])

                embedding_latent = leaky_relu( tf.matmul(state, w) + b )

            with tf.variable_scope("rnn", reuse=None) as rnn_scope:
                rnn_unit_gen = SeqGAN.rnn_unit(
                    self.config.lstm_units_gen, 
                    self.config.lstm_layers_gen,
                    self.config.keep_prob
                    )

                with tf.variable_scope("input_embedding"):
                    wi = tf.get_variable("weight", [self.config.sequence_width + self.config.embedding_latent, self.config.lstm_input_gen])
                    bi = tf.get_variable("bias", [self.config.lstm_input_gen])

                with tf.variable_scope("output_embedding"):
                    wo = tf.get_variable("weight", [self.config.lstm_units_gen, self.config.sequence_width])
                    bo = tf.get_variable("bias", [self.config.sequence_width])

                outputs = []
                rnn_state = rnn_unit_gen.zero_state(self.config.batch_size, dtype=tf.float32)
                rnn_out = tf.zeros([self.config.batch_size, self.config.sequence_width])
                for i in range(self.config.time_steps):
                    # reuse set
                    if i > 0:
                        rnn_scope.reuse_variables()

                    rnn_in = tf.concat([embedding_latent, rnn_out], axis=1)
                    rnn_in = leaky_relu(tf.matmul(rnn_in, wi) + bi)

                    rnn_out, rnn_state = rnn_unit_gen(rnn_in, rnn_state)
                    
                    rnn_out = tf.nn.tanh(tf.matmul(rnn_out, wo) + bo)
                    outputs.append(rnn_out)

                outputs = tf.stack(outputs, axis=1)

        return outputs

    def discriminator_network(self, seq, define=False):

        with tf.variable_scope("discriminator", reuse=(not define)):

            layers_params = [
                ([5, 5, 1, 3], [1, 1, 1, 1]),
                ([5, 5, 3, 5], [1, 1, 1, 1]),
                ([5, 5, 5, 3], [1, 1, 1, 1]),
                ([5, 5, 3, 1], [1, 1, 1, 1])
            ]

            cnn_unit_disc = SeqGAN.cnn_unit(layers_params, "leaky_relu", "SAME")

            seq = tf.reshape(seq, [-1, self.config.time_steps, self.config.sequence_width, 1])
            out_cnn = cnn_unit_disc(seq)
            
            out_pool = tf.reshape(
                tf.nn.avg_pool(
                    out_cnn, 
                    [1, self.config.time_steps, 1, 1],
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

        out_gen = self.out_gen = self.generator_network(self.latent, True)

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

def leaky_relu(data):
    return tf.maximum(data, 0.2*data)