import tensorflow as tf
from .base_model import BaseModel
import pprint

pp = pprint.PrettyPrinter()


class GAN(BaseModel):

    def create_placeholders(self):

        self.image = tf.placeholder(
                dtype=tf.float32,
                shape=[None, 784]
            )

    def generator_network(self, state):

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            w = tf.get_variable("weight", [self.config.latent_state_size, self.config.output_size])
            b = tf.get_variable("bias", [self.config.output_size])

            out = tf.nn.sigmoid( tf.matmul(state, w) + b )

        return out

    def discriminator_network(self, x):

        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            inp = x

            w = tf.get_variable("weight", [self.config.output_size, 1])
            b = tf.get_variable("bias", [1])

            prob = tf.nn.sigmoid( tf.matmul(inp, w) + b )

        return prob

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
        disc_out_target = self.disc_out_target = self.discriminator_network(self.image)

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

