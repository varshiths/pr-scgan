import tensorflow as tf
import numpy as np
from .base_model import BaseModel
import pprint

ppr = pprint.PrettyPrinter()
sprint = lambda x: ppr.pprint(x)

class CSeqGAN(BaseModel):

    def create_placeholders(self):
        # sequence
        self.gesture = tf.placeholder(
                dtype=tf.float32,
                shape=[self.config.batch_size, self.config.sequence_length, self.config.sequence_width, 4],
                name="gesture",
            )

        self.sentence = tf.placeholder(
                dtype=tf.float32,
                shape=[self.config.batch_size, self.config.annot_seq_length, self.config.vocab_size],
                name="sentence",
            )

        self.latent = tf.placeholder(
                dtype=tf.float32,
                shape=[self.config.batch_size, self.config.latent_state_size],
                name="latent",
            )

        self.start = tf.placeholder(
                dtype=tf.float32,
                shape=[self.config.batch_size, self.config.sequence_width, 4],
                name="start",
            )

    def rnn_unit(num_units, num_layers, keep_prob):

        def rnn_cell():
            return tf.contrib.rnn.DropoutWrapper(
            tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=num_units),
            output_keep_prob=keep_prob,
            variational_recurrent=True,
            dtype=tf.float32
            )

        return tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(num_layers)])

    def cfblock(self, inputs):

        with tf.variable_scope("cfblock"):

            lp0 = [
                ([5, 5, 1, 2], [1, 1, 1, 1], [1, 1, 1, 1]),
                ([5, 5, 2, 3], [1, 1, 1, 1], [1, 1, 1, 1]),
            ]
            lp1 = [
                ([5, 5, 3, 2], [1, 1, 1, 1], [1, 1, 1, 1]),
                ([5, 5, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1]),
            ]
            cnn_unit_disc0 = CSeqGAN.cnn_unit(lp0, "leaky_relu", "SAME", "block0")
            cnn_unit_disc1 = CSeqGAN.cnn_unit(lp1, "leaky_relu", "SAME", "block1")

            # expand dims
            inputs = tf.expand_dims(inputs, axis=1)

            # conv + res
            outputs = cnn_unit_disc0(inputs)
            outputs += inputs
            outputs = cnn_unit_disc1(outputs)
            outputs += inputs

            # expand dims
            outputs = tf.squeeze(outputs, axis=1)

        return outputs

    def position_embeddings(self, inputs):

        b, t, dim = inputs.shape.as_list()
        indices = np.arange(t)
        dims = 1.0/np.power(10000.0, (np.arange(dim)/2).astype(np.int32)*2/dim)

        pose = np.matmul(np.expand_dims(indices, 1), np.expand_dims(dims, 0))
        pose[:, 0::2] = np.sin(pose[:, 0::2])
        pose[:, 1::2] = np.cos(pose[:, 1::2])
        pose = np.expand_dims(pose, 0)
        
        return inputs + pose

    def encoder_network(self, sentence):

        with tf.variable_scope("encoder"):
            # embeddings
            # todo: replace with matmul for one hot
            with tf.variable_scope("embeddings"):
                table = tf.layers.Dense(self.config.annot_embedding, name="table")
                embed_inputs = table(sentence)

            # position embeddings
            embed_inputs = self.position_embeddings(embed_inputs)

            # cfblock
            cfblock_outputs = self.cfblock(embed_inputs)

            # rnn
            cells_fw = CSeqGAN.rnn_unit(
                    self.config.lstm_units_enc, 
                    self.config.lstm_layers_enc,
                    self.config.keep_prob
                    )
            cells_bw = CSeqGAN.rnn_unit(
                    self.config.lstm_units_enc, 
                    self.config.lstm_layers_enc,
                    self.config.keep_prob
                    )
            outputs = tf.nn.bidirectional_dynamic_rnn(
                    cells_fw,
                    cells_bw,
                    cfblock_outputs,
                    dtype=tf.float32,
                )
            outputs = tf.concat(outputs[0], axis=2)

        return outputs

    def decoder_network(self, states, latent, start):
        
        with tf.variable_scope("decoder"):
            batch_size = states.shape[0].value

            # concat enc states, latent and embed
            with tf.variable_scope("concat_embedding"):
                latent = tf.expand_dims(latent, axis=1)
                latent = tf.tile(latent, (1, states.shape[1].value, 1))
                sl = tf.concat([states, latent], axis=2)

                embed_states = tf.layers.dense(sl, self.config.embed_states)
                embed_states = tf.contrib.layers.batch_norm(embed_states, is_training=self.config.train_phase)
                embed_states = tf.nn.leaky_relu(embed_states)

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

            # multi head attention mechanism + decode
        
            cell = CSeqGAN.rnn_unit(
                self.config.lstm_units_gen, 
                self.config.lstm_layers_gen,
                self.config.keep_prob
                )
            cell_initial_state = cell.zero_state(batch_size, tf.float32)

            # input embedding
            input_embedding = tf.layers.Dense(self.config.lstm_input_gen, name="input_embedding")

            def apply_attention_net(_input, head):
                att0 = tf.layers.dense(_input, self.config.annot_seq_length*10, activation=tf.nn.leaky_relu, name="head%d/0"%head)
                att1 = tf.layers.dense(att0, 1, name="head%d/1"%head)
                return att1

            def get_context_vec(query, scope="attention", reuse=True):
                with tf.variable_scope(scope, reuse=reuse):
                    query = tf.concat([tup.h for tup in query], axis=1)
                    query = tf.expand_dims(query, axis=1)
                    query = tf.tile(query, [1, embed_states.shape[1].value, 1])
                    states_split = tf.split(embed_states, self.config.att_heads, axis=2)

                    ccq_split = [ tf.concat([query, s], axis=2) for s in states_split ]
                    prob_split = [ tf.nn.softmax(tf.squeeze(apply_attention_net(s, ind), axis=2), axis=1) for ind, s in enumerate(ccq_split) ]

                    ws_split = [ tf.squeeze(tf.matmul(tf.expand_dims(p, axis=1),s), axis=1) for p,s in zip(prob_split, states_split) ]
                    ctxt = tf.concat(ws_split, axis=1)
                return ctxt

            def initialize():
                start_shaped = tf.reshape(start, [batch_size, -1])
                next_inputs = tf.concat([get_context_vec(cell_initial_state, reuse=tf.get_variable_scope().reuse), start_shaped], axis=1)
                next_inputs = input_embedding(next_inputs)
                next_inputs = tf.contrib.layers.batch_norm(
                    next_inputs, 
                    is_training=self.config.train_phase,
                    scope="token_batchnorm",
                    reuse=tf.get_variable_scope().reuse,
                    )
                next_inputs = tf.nn.leaky_relu(next_inputs)
                finished = tf.tile([False], [batch_size])
                return (finished, next_inputs)

            def sample(time, outputs, state):
                samples = tf.tile([0], [batch_size])
                return samples

            def next_inputs(time, outputs, state, sample_ids):
                
                next_inputs = tf.concat([get_context_vec(state), outputs], axis=1)
                next_inputs = input_embedding(next_inputs)
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
                initial_state=cell_initial_state,
                output_layer=tf.layers.Dense(
                        units=self.config.sequence_width * 4,
                        activation=quart_activation,
                        kernel_initializer=None,
                        bias_initializer=None,
                        name="output_embedding",
                    ),
                )
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                parallel_iterations=12,
                maximum_iterations=self.config.sequence_length)

            outputs = outputs.rnn_output

            outputs = tf.reshape(outputs, [batch_size, self.config.sequence_length, self.config.sequence_width, 4])

        return outputs

    def generator_network(self, sentence, latent, start, define=False):

        with tf.variable_scope("generator", reuse=(not define)):
           
           out_enc = self.encoder_network(sentence)
           outputs = self.decoder_network(out_enc, latent, start)

        if define:
            self.gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        return outputs

    def cnn_unit(params, activation=None, padding="SAME", scope=""):

        def get_variable(filt_dim, scope):
            return tf.get_variable(scope, filt_dim)

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

        seq = tf.reshape(seq, [-1, self.config.sequence_length, self.config.sequence_width * 4])

        with tf.variable_scope("discriminator", reuse=(not define)):

            batch_size = seq.shape[0].value

            # Convolution with pooling
            # Features are not spacially correlated
            out_cnn = tf.reshape(seq, [-1, 1, self.config.sequence_length, self.config.sequence_width*4])

            lp0 = [([10, 4,  1, 4 * 10], [1, 1, 2, 4], [1, 1, 2, 1]),]
            lp1 = [([10, 4, 10, 4 * 10], [1, 1, 2, 4], [1, 1, 2, 1]),]
            lp2 = [([10, 4, 10, 4 * 10], [1, 1, 2, 4], [1, 1, 2, 1]),]
            lp3 = [([10, 4, 10,      1], [1, 1, 2, 4], [1, 1, 2, 1]),]
            cnn_unit_disc0 = CSeqGAN.cnn_unit(lp0, "leaky_relu", "SAME", "block0")
            cnn_unit_disc1 = CSeqGAN.cnn_unit(lp1, "leaky_relu", "SAME", "block1")
            cnn_unit_disc2 = CSeqGAN.cnn_unit(lp2, "leaky_relu", "SAME", "block2")
            cnn_unit_disc3 = CSeqGAN.cnn_unit(lp3, "leaky_relu", "SAME", "block3")

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
                out = tf.layers.dense(out_pool, 64, activation=tf.nn.leaky_relu)
                out = tf.layers.dense(out, 32, activation=tf.nn.leaky_relu)
                out = tf.layers.dense(out, 1)
                out = tf.reshape(out, [-1])

        if define:
            self.disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
        return out

    def generator_costs(self, discval_gen, _gen, _target):

        # discriminator cost
        disc_cost = -tf.reduce_mean(discval_gen)
        # pretrain_cost
        pretrain_cost = self.generator_pretrain_cost(_gen, _target)
        return pretrain_cost, disc_cost + pretrain_cost

    def generator_pretrain_cost(self, _gen, _target):

        # smoothness
        differences = tf.square(_gen[:,1:,:,:]-_gen[:,:-1,:,:])
        smoothness_cost = tf.maximum(tf.reduce_mean(differences)-self.config.smoothness_threshold, 0.0)

        # supervision cost
        # todo: dtw
        target_cost = tf.reduce_mean(tf.square(_gen - _target))

        # accumulated cost
        cost = target_cost + smoothness_cost*self.config.smoothness_weight
        return cost

    def discriminator_cost(self, discval_gen, discval_target, _gen, _target):

        batch_size = _target.shape[0].value

        with tf.variable_scope("cost", reuse=False):
            ep = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
            x = tf.get_variable("temp", [batch_size, self.config.sequence_length, self.config.sequence_width, 4], trainable=False)
            x = tf.assign(x, _target*ep + _gen*(1-ep))

        disc_x = self.discriminator_network(x)
        grads = tf.reshape(tf.gradients(disc_x, x)[0], [batch_size, -1])
        grad_cost = tf.reduce_mean(tf.square(tf.norm(grads, axis=1) - 1))

        cost = tf.reduce_mean(discval_gen)-tf.reduce_mean(discval_target) + \
                grad_cost*self.config.grad_penalty_weight
        return cost

    def build_model(self):

        with tf.variable_scope(tf.get_variable_scope()):

            self.epsilon = tf.constant(1e-8)
            self.create_placeholders()

            # split feed into batches for ngpus
            gesture_a = make_batches(self.config.ngpus, self.gesture)
            sentence_a = make_batches(self.config.ngpus, self.sentence)
            latent_a = make_batches(self.config.ngpus, self.latent)
            start_a = make_batches(self.config.ngpus, self.start)
            
            out_gen_list = []
            gen_cost_list = []
            gen_pretrain_cost_list = []
            disc_cost_list = []
            
            gen_pretrain_grads_list = []
            gen_grads_list = []
            disc_grads_list = []

            for i in range(self.config.ngpus):

                gesture, sentence, latent, start = gesture_a[i], sentence_a[i], latent_a[i], start_a[i]

                with tf.device('/gpu:%d' % i):
                    # import pdb; pdb.set_trace()
                    out_gen = self.generator_network(sentence, latent, start, i==0)

                    disc_out_gen = self.discriminator_network(out_gen, i==0)
                    disc_out_target = self.discriminator_network(gesture)

                    gen_pretrain_cost, gen_cost = self.generator_costs(disc_out_gen, out_gen, gesture)
                    disc_cost = self.discriminator_cost(disc_out_gen, disc_out_target, out_gen, gesture)

                    gen_pretrain_grads = self.compute_and_clip_gradients(gen_pretrain_cost, self.gen_vars)
                    gen_grads = self.compute_and_clip_gradients(gen_cost, self.gen_vars)
                    disc_grads = self.compute_and_clip_gradients(disc_cost, self.disc_vars)

                out_gen_list.append(out_gen)
                gen_pretrain_cost_list.append(gen_pretrain_cost)
                gen_cost_list.append(gen_cost)
                disc_cost_list.append(disc_cost)
                gen_pretrain_grads_list.append(gen_pretrain_grads)
                gen_grads_list.append(gen_grads)
                disc_grads_list.append(disc_grads)

            gen_pretrain_grads = average_gradients(gen_pretrain_grads_list)
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
            self.gen_pretrain_grad_step = optimizer_gen.apply_gradients(zip( gen_pretrain_grads, self.gen_vars ))
            self.gen_grad_step = optimizer_gen.apply_gradients(zip( gen_grads, self.gen_vars ))
            self.disc_grad_step = optimizer_disc.apply_gradients(zip( disc_grads, self.disc_vars ))

            # setting other informative fetches
            self.out_gen = tf.stack(out_gen, axis=0)
            self.gen_pretrain_cost = tf.reduce_mean(tf.convert_to_tensor(gen_pretrain_cost))
            self.gen_cost = tf.reduce_mean(tf.convert_to_tensor(gen_cost))
            self.disc_cost = tf.reduce_mean(tf.convert_to_tensor(disc_cost))
            self.gen_pretrain_grads = tf.reduce_mean([tf.norm(x) for x in gen_pretrain_grads])
            self.gen_grads = tf.reduce_mean([tf.norm(x) for x in gen_grads])
            self.disc_grads = tf.reduce_mean([tf.norm(x) for x in disc_grads])

            self.build_validation_metrics()

    def compute_and_clip_gradients(self, cost, _vars):

        grads = tf.gradients(cost, _vars)
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.config.max_grad)
        return clipped_grads

    def build_validation_metrics(self):

        sprint(self.gen_vars)
        sprint(self.disc_vars)

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
        batched = tf.split(data, n, axis=0)
    except Exception as e:
        raise Exception("Batch size not a multiple of ngpus")
    return batched
