import tensorflow as tf
from .base_model import BaseModel


class SGAN(BaseModel):

    def init_saver(self):
        # just copy the following line in your child class
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        self.epsilon = tf.constant(1e-8)

    def create_placeholders(self):

        self.label = tf.placeholder(
                dtype=tf.float32,
                shape=[None, 10]
            )

        self.image = tf.placeholder(
                dtype=tf.float32,
                shape=[None, 784]
            )

        self.real = tf.placeholder(
                dtype=tf.float32,
                shape=[None]
            )

    def create_embedding(self):

        with tf.variable_scope("embedding"):
            w = tf.get_variable("weight", [10, self.config.embedding])
            b = tf.get_variable("bias", [self.config.embedding])

            embed = tf.nn.softmax(tf.matmul( self.label, w ) + b)

        return embed

    def inference_network(self, inp):

        with tf.variable_scope("tins"):
            w = tf.get_variable("weight", [self.config.embedding, self.config.internal_state])
            b = tf.get_variable("bias", [self.config.internal_state])

            output = tf.nn.softmax( tf.matmul( inp, w ) + b )
        
        return output

    def get_generator_network(self):

        with tf.variable_scope("generator"):
            w = tf.get_variable("weight", [self.config.internal_state, 784])
            b = tf.get_variable("bias", [784])

            def function(state):
                image = tf.nn.softmax( tf.matmul(state, w) + b )
                return image

        return function

    def get_discriminator_network(self):

        with tf.variable_scope("discriminator"):
            w = tf.get_variable("weight", [784, 1])
            b = tf.get_variable("bias", [1])

            def function(image):
                prob = tf.nn.softmax( tf.matmul(image, w) + b )
                return prob

        return function

    def generator_cost(self, prob_image_gen):

        logt = tf.log( 1 - prob_image_gen + self.epsilon )
        cost = tf.reduce_mean(logt)
        return cost

    def discriminator_cost(self, prob_image_gen, prob_image_target):

        logtg = tf.log( 1 - prob_image_gen + self.epsilon )
        logtt = tf.log( prob_image_target + self.epsilon )

        cost = tf.reduce_mean( logtt + logtg )
        return cost

    def build_model(self):

        generator_network = self.get_generator_network()
        discriminator_network = self.get_discriminator_network()

        self.create_placeholders()

        in_embed = self.create_embedding()
        internal_state = self.inference_network(in_embed)

        image_gen = generator_network(internal_state)

        disc_image_gen = discriminator_network(image_gen)
        disc_image_target = discriminator_network(self.image)

        self.gen_cost = self.generator_cost(disc_image_gen)
        self.disc_cost = self.discriminator_cost(disc_image_gen, disc_image_target)

