import tensorflow as tf
from .base_model import BaseModel
import pprint

pp = pprint.PrettyPrinter()


class SGAN(BaseModel):

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
            w = tf.get_variable("weight", [self.config.input_size, self.config.embedding])
            b = tf.get_variable("bias", [self.config.embedding])

            embed = tf.nn.sigmoid(tf.matmul( inp, w ) + b)

        return embed

    def inference_network(self, inp):

        with tf.variable_scope("tins", reuse=tf.AUTO_REUSE):
            w = tf.get_variable("weight", [self.config.embedding, self.config.internal_state])
            b = tf.get_variable("bias", [self.config.internal_state])

            output = tf.nn.sigmoid( tf.matmul( inp, w ) + b )
        
        return output

    def generator_network(self, state):

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            w = tf.get_variable("weight", [self.config.internal_state, self.config.output_size])
            b = tf.get_variable("bias", [self.config.output_size])

            image = tf.nn.softmax( tf.matmul(state, w) + b )

        return image

    def discriminator_network(self, image):

        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            w = tf.get_variable("weight", [self.config.output_size + self.config.input_size, 1])
            b = tf.get_variable("bias", [1])

            prob = tf.nn.sigmoid( tf.matmul(image, w) + b )

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

        in_embed = self.create_embedding(self.image)
        internal_state = self.inference_network(in_embed)

        out_gen = self.out_gen = self.generator_network(internal_state)

        disc_in_gen = tf.concat([self.image, out_gen], axis=1)
        disc_in_target = tf.concat([self.image, self.label], axis=1)

        disc_out_gen = self.disc_out_gen = self.discriminator_network(disc_in_gen)
        disc_out_target = self.disc_out_target = self.discriminator_network(disc_in_target)

        self.gen_cost = self.generator_cost(disc_out_gen)
        self.disc_cost = self.discriminator_cost(disc_out_gen, disc_out_target)

        self.build_gradient_steps()

        self.build_validation_metrics()

    def build_gradient_steps(self):

        emb_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="embedding")
        tins_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="tins")
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

        gen_grad_vars = emb_vars + tins_vars + gen_vars
        disc_grad_vars = emb_vars + tins_vars + gen_vars + disc_vars

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

        eq_bools = tf.equal(tf.argmax(self.out_gen,1), tf.argmax(self.label,1))
        self.validation_error = 100.0 * tf.reduce_mean( tf.cast(eq_bools, dtype=tf.float32) )

