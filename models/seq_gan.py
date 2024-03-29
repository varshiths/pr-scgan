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
                shape=[self.config.batch_size, self.config.time_steps, self.config.sequence_width, 4]
            )

        self.latent = tf.placeholder(
                dtype=tf.float32,
                shape=[self.config.batch_size, self.config.latent_state_size]
            )

        self.start = tf.placeholder(
                dtype=tf.float32,
                shape=[self.config.batch_size, self.config.sequence_width, 4]
            )
        self.start_token = tf.random_normal((self.config.batch_size, self.config.sequence_width, 4))

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

                embedding_latent = tf.matmul(state, w) + b
                embedding_latent = tf.contrib.layers.batch_norm(embedding_latent, is_training=self.config.train_phase)
                embedding_latent = tf.nn.leaky_relu(embedding_latent)

            def quart_activation(inp):

                actv = tf.reshape(inp, [-1, 4])
                actv = tf.contrib.layers.batch_norm(actv, is_training=self.config.train_phase)
                # make updates to first column of quart
                actv = tf.concat([
                        tf.nn.sigmoid(actv[:, :1]),
                        tf.nn.tanh(actv[:, 1:]),
                    ], axis=-1)
                # normalise for rot quarts
                actv = actv / tf.norm(actv, axis=-1, keepdims=True)
                # reshape for feed
                actv = tf.reshape(actv, [batch_size, -1])

                return actv

            with tf.variable_scope("rnn", reuse=None) as rnn_scope:
                cell = SeqGAN.rnn_unit(
                    self.config.lstm_units_gen, 
                    self.config.lstm_layers_gen,
                    self.config.keep_prob
                    )

                with tf.variable_scope("input_embedding"):
                    wi = tf.get_variable("weight", [self.config.embedding_latent+self.config.sequence_width*4, self.config.lstm_input_gen])
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
                    start_shaped = tf.reshape(start, [batch_size, -1])
                    next_inputs = tf.concat([embedding_latent, start_shaped], axis=1)
                    next_inputs = tf.matmul(next_inputs, wi) + bi
                    next_inputs = tf.contrib.layers.batch_norm(
                        next_inputs, 
                        is_training=self.config.train_phase,
                        scope="token_batchnorm",
                        reuse=False,
                        )
                    next_inputs = tf.nn.leaky_relu(next_inputs)
                    finished = tf.tile([False], [batch_size])
                    return (finished, next_inputs)

                def sample(time, outputs, state):
                    samples = tf.tile([0], [batch_size])
                    return samples

                def next_inputs(time, outputs, state, sample_ids):
                    
                    next_inputs = tf.concat([embedding_latent, outputs], axis=1)
                    next_inputs = tf.matmul(next_inputs, wi) + bi
                    next_inputs = tf.contrib.layers.batch_norm(
                        next_inputs, 
                        is_training=self.config.train_phase,
                        scope="token_batchnorm",
                        reuse=True,
                        )
                    next_inputs = tf.nn.leaky_relu(next_inputs)
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
                            units=self.config.sequence_width * 4,
                            activation=quart_activation,
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

            outputs = tf.reshape(outputs, [batch_size, self.config.time_steps, self.config.sequence_width, 4])

        if define:
            self.gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        return outputs

    def cnn_unit(params, activation=None, padding="SAME", scope=""):

        def get_variable(filt_dim, scope):
            return tf.get_variable(scope + "/filter", filt_dim)

        def function(inp):
            out = inp
            for i, (filt_dim, strides, dilations) in enumerate(params):
                out = tf.nn.conv2d(
                    out, 
                    get_variable(filt_dim, scope+"/conv%d" % (i)), 
                    data_format="NCHW",
                    strides=strides,
                    dilations=dilations,
                    padding=padding
                    )
                if activation is not None:
                    out = tf.nn.leaky_relu(out)
            return out

        return function

    def discriminator_network(self, seq, define=False):

        seq = tf.reshape(seq, [-1, self.config.time_steps, self.config.sequence_width * 4])

        with tf.variable_scope("discriminator", reuse=(not define)):

            # import pdb
            # pdb.set_trace()

            batch_size = seq.shape[0].value

            # Convolution with pooling
            # Features are not spacially correlated
            out_cnn = tf.reshape(seq, [-1, 1, self.config.time_steps, self.config.sequence_width*4])

            lp0 = [([10, 4,  1, 4 * 10], [1, 1, 2, 4], [1, 1, 2, 1]),]
            lp1 = [([10, 4, 10, 4 * 10], [1, 1, 2, 4], [1, 1, 2, 1]),]
            lp2 = [([10, 4, 10, 4 * 10], [1, 1, 2, 4], [1, 1, 2, 1]),]
            lp3 = [([10, 4, 10,      1], [1, 1, 2, 4], [1, 1, 2, 1]),]
            cnn_unit_disc0 = SeqGAN.cnn_unit(lp0, "leaky_relu", "SAME", "block0")
            cnn_unit_disc1 = SeqGAN.cnn_unit(lp1, "leaky_relu", "SAME", "block1")
            cnn_unit_disc2 = SeqGAN.cnn_unit(lp2, "leaky_relu", "SAME", "block2")
            cnn_unit_disc3 = SeqGAN.cnn_unit(lp3, "leaky_relu", "SAME", "block3")

            out_cnn = cnn_unit_disc0(out_cnn)
            out_cnn = tf.transpose(
                tf.reshape(out_cnn, [batch_size, 10, 4, out_cnn.shape[2].value, 107]), (0,1,3,4,2))
            out_cnn = tf.reshape(out_cnn, [batch_size, 10, -1, 428])

            out_cnn = cnn_unit_disc1(out_cnn)
            out_cnn = tf.transpose(
                tf.reshape(out_cnn, [batch_size, 10, 4, out_cnn.shape[2].value, 107]), (0,1,3,4,2))
            out_cnn = tf.reshape(out_cnn, [batch_size, 10, -1, 428])

            out_cnn = cnn_unit_disc2(out_cnn)
            out_cnn = tf.transpose(
                tf.reshape(out_cnn, [batch_size, 10, 4, out_cnn.shape[2].value, 107]), (0,1,3,4,2))
            out_cnn = tf.reshape(out_cnn, [batch_size, 10, -1, 428])

            out_cnn = cnn_unit_disc3(out_cnn)
            
            out_pool = tf.reshape(
                tf.nn.avg_pool(
                    out_cnn, 
                    [1, 1, out_cnn.shape[2].value, 1],
                    strides=[1,1,1,1],
                    padding="VALID",
                    ), 
                [-1, self.config.sequence_width]
                )
            
            with tf.variable_scope("prob_embed"):
                w0 = tf.get_variable("weight0", [self.config.sequence_width, 50])
                b0 = tf.get_variable("bias0", [50])

                w1 = tf.get_variable("weight1", [50, 1])
                b1 = tf.get_variable("bias1", [1])

                out = tf.nn.leaky_relu( tf.matmul(out_pool, w0) + b0 )
                out = tf.matmul(out, w1) + b1
                out = tf.reshape(out, [-1])

            # # Fully connected network

            '''
            # time independent compression of each sequence
            out = tf.reshape(seq, [-1, self.config.sequence_width*4])
            out = tf.layers.dense(
                    inputs=out,
                    units=300,
                    activation=None,
                    kernel_initializer=tf.random_normal_initializer(),
                    bias_initializer=tf.random_normal_initializer(),
                )
            out = tf.contrib.layers.batch_norm(out, is_training=self.config.train_phase, activation_fn=tf.nn.leaky_relu)
            out = tf.layers.dense(
                    inputs=out,
                    units=100,
                    activation=None,
                    kernel_initializer=tf.random_normal_initializer(),
                    bias_initializer=tf.random_normal_initializer(),
                )
            out = tf.contrib.layers.batch_norm(out, is_training=self.config.train_phase, activation_fn=tf.nn.leaky_relu)

            # ff across time
            out = tf.reshape(out, [batch_size, -1])
            out = tf.layers.dense(
                    inputs=out,
                    units=1000,
                    activation=None,
                    kernel_initializer=tf.random_normal_initializer(),
                    bias_initializer=tf.random_normal_initializer(),
                )
            out = tf.contrib.layers.batch_norm(out, is_training=self.config.train_phase, activation_fn=tf.nn.leaky_relu)
            out = tf.layers.dense(
                    inputs=out,
                    units=1,
                    activation=None,
                    kernel_initializer=tf.random_normal_initializer(),
                    bias_initializer=tf.random_normal_initializer(),
                )
            '''

        if define:
            self.disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
        return out

    def generator_cost(self, discval_gen, _gen):

        batch_size = discval_gen.shape[0].value

        differences = tf.reshape(_gen[:,1:,:,:]-_gen[:,:-1,:,:], [batch_size, -1])
        diff_cost = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(differences), axis=1)))
        cost = -tf.reduce_mean(discval_gen) + self.config.smoothness_weight*diff_cost
        return cost

    def discriminator_cost(self, discval_gen, discval_target, _gen, _target):

        batch_size = _target.shape[0].value

        ep = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
        with tf.variable_scope("cost", reuse=tf.AUTO_REUSE):
            x = tf.get_variable("temp", [batch_size, self.config.time_steps, self.config.sequence_width, 4], trainable=False)
            x = tf.assign(x, _target*ep + _gen*(1-ep))
        # x = tf.Variable(_target*ep + _gen*(1-ep))

        disc_x = self.discriminator_network(x)
        grads = tf.reshape(tf.gradients(disc_x, x)[0], [batch_size, -1])
        grad_error = tf.reduce_mean(tf.square(tf.norm(grads, axis=1) - 1))

        cost = tf.reduce_mean(discval_gen)-tf.reduce_mean(discval_target) + \
                self.config.grad_penalty_weight*tf.reduce_mean(grad_error)
        return cost

    def build_model(self):

        self.epsilon = tf.constant(1e-8)
        self.create_placeholders()

        # split feed into batches for ngpus
        data_a = make_batches(self.config.ngpus, self.data)
        latent_a = make_batches(self.config.ngpus, self.latent)
        start_a = make_batches(self.config.ngpus, self.start)
        
        out_gen_list = []
        gen_cost_list = []
        disc_cost_list = []
        
        gen_grads_list = []
        disc_grads_list = []

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.config.ngpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ("TOWER", i)) as scope:

                        data, latent, start = data_a[i], latent_a[i], start_a[i] 

                        out_gen = self.generator_network(latent, start, i==0)

                        disc_out_gen = self.discriminator_network(out_gen, i==0)
                        disc_out_target = self.discriminator_network(data)

                        gen_cost = self.generator_cost(disc_out_gen, out_gen)
                        disc_cost = self.discriminator_cost(disc_out_gen, disc_out_target, out_gen, data)

                        gen_grads = self.compute_and_clip_gradients(gen_cost, self.gen_vars)
                        disc_grads = self.compute_and_clip_gradients(disc_cost, self.disc_vars)

                        out_gen_list.append(out_gen)
                        gen_cost_list.append(gen_cost)
                        disc_cost_list.append(disc_cost)
                        gen_grads_list.append(gen_grads)
                        disc_grads_list.append(disc_grads)

        gen_grads = average_gradients(gen_grads_list)
        disc_grads = average_gradients(disc_grads_list)

        # defining optimizer
        optimizer_gen = tf.train.AdamOptimizer(
            learning_rate=self.config.learning_rate
        )

        optimizer_disc = tf.train.AdamOptimizer(
            learning_rate=self.config.learning_rate/2
        )

        # main step fetches
        self.gen_grad_step = optimizer_gen.apply_gradients(zip( gen_grads, self.gen_vars ))
        self.disc_grad_step = optimizer_disc.apply_gradients(zip( disc_grads, self.disc_vars ))

        # setting other informative fetches
        self.out_gen = tf.stack(out_gen, axis=0)
        self.gen_cost = tf.reduce_mean(tf.convert_to_tensor(gen_cost))
        self.disc_cost = tf.reduce_mean(tf.convert_to_tensor(disc_cost))
        self.gen_grads = tf.reduce_mean([tf.norm(x) for x in gen_grads])
        self.disc_grads = tf.reduce_mean([tf.norm(x) for x in disc_grads])

        self.build_validation_metrics()

    def compute_and_clip_gradients(self, cost, _vars):

        grads = tf.gradients(cost, _vars)
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.config.max_grad)
        return clipped_grads

    def build_validation_metrics(self):

        pp.pprint(self.gen_vars)
        pp.pprint(self.disc_vars)

        pass

def average_gradients(grads):

    average_grads = []
    for gradv in zip(*grads):
      # Average over the 'tower' dimension.
      grad = tf.stack(gradv, axis=0)
      grad = tf.reduce_mean(grad, axis=0)

      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower's pointer to
      # the Variable.
      average_grads.append(grad)
    return average_grads

def make_batches(n, data):
    try:
        shape = [x.value for x in list(data.shape)]
        shape[0] = int(shape[0]/n)
        shape.insert(0, n)
        batched = tf.reshape(data, shape)
    except Exception as e:
        raise Exception("Batch size not a multiple of ngpus")
    return batched
