import tensorflow as tf
from .base_model import BaseModel
import pprint

pp = pprint.PrettyPrinter()


class GAN(BaseModel):

    def create_placeholders(self):

        self.data = tf.placeholder(
                dtype=tf.float32,
                shape=[None, 784]
            )

        self.latent = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.config.latent_state_size]
            )

    def generator_network(self, state):

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            w0 = tf.get_variable("weight0", [self.config.latent_state_size, self.config.mid_dim])
            b0 = tf.get_variable("bias0", [self.config.mid_dim])

            out = tf.nn.leaky_relu( tf.matmul(state, w0) + b0 )

            w1 = tf.get_variable("weight1", [self.config.mid_dim, self.config.output_size])
            b1 = tf.get_variable("bias1", [self.config.output_size])

            out = tf.nn.tanh( tf.matmul(out, w1) + b1 )

        return out

    def discriminator_network(self, x):

        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            inp = x

            w0 = tf.get_variable("weight0", [self.config.output_size, self.config.mid_dim])
            b0 = tf.get_variable("bias0", [self.config.mid_dim])

            out = tf.nn.leaky_relu( tf.matmul(inp, w0) + b0 )

            w1 = tf.get_variable("weight1", [self.config.mid_dim, 1])
            b1 = tf.get_variable("bias1", [1])

            out = tf.nn.sigmoid( tf.matmul(out, w1) + b1 )

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

        out_gen = self.out_gen = self.generator_network(self.latent)

        disc_out_gen = self.disc_out_gen = self.discriminator_network(out_gen)
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

