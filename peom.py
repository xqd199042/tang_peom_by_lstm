import numpy
from my_framework_multi_gpu import Config, Tensors, App
from qts import QTS
import tensorflow as tf


class MyConfig(Config):

    def __init__(self):
        super(MyConfig, self).__init__()
        self.num_units = 300
        self.num_steps = 32
        self.batch_size = 300
        self.epoches = 2000
        self.drop_out_keep_ratio = 0.6
        self.lr = 0.0001
        self.num_chars = 0

    def get_data(self):
        qts = QTS('qts.txt')
        self.num_chars = len(qts.chars)
        return qts

    def get_tensors(self):
        return MyTensors(self)

    def get_name(self):
        return 'p69_peom_multi_rnn'


class MyTensors(Tensors):
    '''
    this class is to override Tensors
    to construct forward process
    '''
    def __init__(self, config:MyConfig):
        super(MyTensors, self).__init__(config)

    def get_sub_tensors(self, index):
        result={}
        x = tf.placeholder(tf.int64, [None, self.config.num_steps], 'x_'+index)
        result['inputs'] = [x]
        x = tf.one_hot(x, self.config.num_chars)
        y = tf.layers.dense(x, self.config.num_units, name='dense_enc')

        #lstm of 2 layers
        cell1 = tf.nn.rnn_cell.LSTMCell(self.config.num_units, name='cell1', state_is_tuple=False)
        cell2 = tf.nn.rnn_cell.LSTMCell(self.config.num_units, name='cell2', state_is_tuple=False)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2], state_is_tuple=False)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.config.drop_out_keep_ratio)

        #define a state memory
        state = cell.zero_state(tf.shape(y)[0], y.dtype)
        losses = []
        for i in range(self.config.num_steps):
            yi, state = cell(y[:, i, :], state)
            output = tf.layers.dense(yi, self.config.num_chars, name='dense_dec')
            if i < self.config.num_steps-1:
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=x[:, i+1, :], logits=output)
                losses.append(loss)
        result['loss'] = [tf.reduce_mean(losses)]

        #tensors below are for testing
        x_i = tf.placeholder(tf.int64, [None], 'xi')
        zero_state_i = cell.zero_state(tf.shape(x_i)[0], y.dtype)
        result['x_i'] = x_i
        result['zero_state'] = zero_state_i
        x_i = tf.one_hot(x_i, self.config.num_chars)
        x_i = tf.layers.dense(x_i, self.config.num_units, name='dense_enc')
        state = tf.placeholder(tf.float32, [None, 4*self.config.num_units])
        result['state'] = state
        y_i, state_i = cell(x_i, state)
        result['state_i'] = state_i
        y_i = tf.layers.dense(y_i, self.config.num_chars, name='dense_dec')
        result['y_predict'] = tf.argmax(y_i, axis=1)

        return result


class MyApp(App):

    def test(self):
        qts = self.ds
        num_chars = self.config.num_chars
        ts = self.ts.sub_ts[-1]
        x_i = numpy.random.randint(0, num_chars, [5, 1])
        result = [x_i[:, 0]]
        state = self.session.run(ts['zero_state'], {ts['x_i']: x_i[:, 0]})
        for i in range(self.config.num_steps-1):
            fd = {ts['x_i']: x_i[:, 0], ts['state']: state} if i==0 else {ts['x_i']: x_i, ts['state']: state}
            x_i, state = self.session.run([ts['y_predict'], ts['state_i']], fd)
            result.append(x_i)
        result = numpy.transpose(result, [1, 0])
        for i in range(len(result)):
            print(qts.get_chars(*list(result[i, :])))


if __name__ == '__main__':
    cfg = MyConfig()
    app = MyApp(cfg)
    with app:
        app.test()