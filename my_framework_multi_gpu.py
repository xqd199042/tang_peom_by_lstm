import os
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from argparse import ArgumentParser
import tensorflow as tf


class Config:
    '''
    Config类用来存放lr、epoch、batch_size、gpu等配置信息
    可以新建MyConfig类继承Config来更改、添加配置信息
    '''
    def __init__(self):
        self.lr = 0.001
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.epoches = 200
        self.batch_size = 200
        #tensorboard logdir
        self.logdir = os.sep.join([self.get_path_prefix(),'Logs','%s']) % self.get_name()
        #model saved dir
        self.savedir = os.sep.join([self.get_path_prefix(), 'Models', '%s', '%s']) % (self.get_name(), self.get_name())
        #if any generated picture, dir
        self.imgdir = os.sep.join([self.get_path_prefix(), 'Pictures', '%s.jpg']) % self.get_name()
        self.gpu_frac = 0.95
        #是否重新训练模型
        self.new_model = False
        #计算momentom的decay
        self.decay = 0.99
        self.dropout_keep_rate = 0.6
        #使用哪几块GPU
        self.gpu = '0'
    #override __repr__方法，
    #打印config对象所有配置信息
    def __repr__(self):
        attrs = self.get_attrs()
        return '\n'.join(['%s=%s'%(attr, attrs[attr]) for attr in attrs])
    #返回config对象中所有attributes的值
    def get_attrs(self):
        attrs = {}
        for attr in dir(self):
            if attr.startswith('_'):
                continue
            value = getattr(self, attr)
            if value is None or type(value) in (int, float, str, bool):
                attrs[attr] = value
        return attrs
    #在服务器端运行模型的时候，
    #该方法用于在terminal中动态更改配置信息
    def get_cmd(self):
        parser = ArgumentParser()
        attrs = self.get_attrs()
        for attr in attrs:
            value = attrs[attr]
            t = type(value)
            if t == bool:
                parser.add_argument('--'+attr, default=value, action='store_%s'%('true' if not value else 'false'))
            else:
                parser.add_argument('--'+attr, type=t, default=value)
        parser.add_argument('--call', type=str, default='train')
        a = parser.parse_args()
        for attr in attrs:
            setattr(self, attr, getattr(a, attr))
        self.get_call(a.call)

    def get_name(self):
        return 'default'

    def get_path_prefix(self):
        return 'D:'

    def get_call(self, a_call):
        pass

    def get_tensors(self):
        return None

    def get_data(self):
        ds = read_data_sets('/home/qiangde/Data/mnist')
        return ds


class Tensors:
    '''
    该类用来描述前向传播过程，
    可以新建MyTensors类继承Tensors来构建前向传播
    '''
    def __init__(self, config: Config):
        self.config = config
        self.sub_ts = []
        with tf.variable_scope(config.get_name(), reuse=tf.AUTO_REUSE):
            for i in config.gpu.split(','):
                with tf.device('/gpu:%s' % i):
                    #在每个GPU上构建前向传播图sub_tensors（sub_ts）
                    self.sub_ts.append(self.get_sub_tensors(i))
        with tf.variable_scope('%s_train' % self.config.get_name()):
            with tf.device('/gpu:0'):
                #在GPU0上平均各个GPU上的梯度并反向传播
                losses = [ts['losses'] for ts in self.sub_ts]
                loss = tf.reduce_mean(losses, axis=0)
                #summary用于tensorboard上查看训练进程，
                #可以override get_loss_for_summary方法来构建不同的指标
                for i in range(len(losses[0])):
                    tf.summary.scalar('loss_%d'%i, self.get_loss_for_summary(loss, i))
                self.summary = tf.summary.merge_all()
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    gradients = [ts['gradients'] for ts in self.sub_ts]
                    grads = []
                    for i in range(len(gradients[0])):
                        grads.append(self.average_grads([gs[i] for gs in gradients]))
                    self.opt = [self.config.optimizer.apply_gradients(grad) for grad in grads]
        #overr get_other_tensors方法来添加一些额外的tensors
        self.get_other_tensors()

    def get_other_tensors(self):
        return None

    def get_loss_for_summary(self, loss, i):
        return loss[i]
    #多GPU上计算平均梯度的方法
    def average_grads(self, grads):
        # grads = (g_gpu_0, g_gpu_1, ..., g_gpu_N),
        # g_gpu_n = ((g_0_gpu_n, var_0), (g_1_gpu_n, var_1), ..., (g_M_gpu_n, var_M))
        average_grads = []
        for grads_and_vars in zip(*grads):
            # zip(*grads) = (((g_0_gpu_o, var_0), (g_0_gpu_1, var_0), ..., (g_0_gpu_N, var_0)),
            #                ((g_1_gpu_0, var_1), (g_1_gpu_1, var_1), ..., (g_1_gpu_N, var_1)),
            #                ...,
            #                ((g_M_gpu_0, var_M), (g_M_gpu_1, var_M), ..., (g_M_gpu_N, var_M)))
            grad = grads_and_vars[0][0]
            var = grads_and_vars[0][1]
            if isinstance(grad, tf.IndexedSlices):
                slices = [(value/len(grads_and_vars), i) for value,i in [(pair[0].values, pair[0].indices) for pair in grads_and_vars]]
                values, indices = zip(*slices)
                values = tf.concat(values, axis=0)
                indices = tf.concat(indices, axis=0)
                average_grads.append((tf.IndexedSlices(values, indices), var))
            else:
                average_grads.append((tf.reduce_mean([g for g,_ in grads_and_vars], axis=0), var))
        return average_grads

    def get_sub_tensors(self, index):
        return None


class App:
    '''
    该类是训练和测试的框架类，
    可以新建MyApp类继承App来自定义自己想要的训练过程
    '''
    def __init__(self, config:Config):
        self.config = config
        self.ds = self.config.get_data()
        print('Data Ready.')
        graph = tf.Graph()
        with graph.as_default():
            self.ts = self.config.get_tensors()
            print('Tensor Ready.')
            cfg = tf.ConfigProto()
            cfg.allow_soft_placement = True
            cfg.gpu_options.allow_growth = True
            cfg.gpu_options.per_process_gpu_memory_fraction = self.config.gpu_frac
            self.session = tf.Session(config=cfg, graph=graph)
            self.saver = tf.train.Saver()
            if self.config.new_model:
                self.session.run(tf.global_variables_initializer())
                print('Create and Use a new Model!')
            else:
                try:
                    self.saver.restore(self.session, self.config.savedir)
                    print('Restore Model from %s Successfully!' % self.config.savedir)
                except Exception as e:
                    print('Fail to Restore Model, Create and Use a New Model!')
                    self.session.run(tf.global_variables_initializer())

    def train(self):
        self.before_train()
        writer = tf.summary.FileWriter(self.config.logdir, self.session.graph)
        batches = self.ds.train.num_examples // (self.config.batch_size*len(self.config.gpu.split(',')))
        for epoch in range(self.config.epoches):
            self.before_epoch(epoch)
            for batch in range(batches):
                feed = self.get_feed_dict()
                if len(self.ts.opt) == 1:
                    _, summary = self.session.run([self.ts.opt[0], self.ts.summary], feed)
                else:
                    for opt in self.ts.opt:
                        self.session.run(opt, feed)
                    summary = self.session.run(self.ts.summary, feed)
                writer.add_summary(summary, epoch*batches+batch)
                self.each_batch(batch)
            self.after_epoch(epoch)
        self.after_train()

    def before_train(self):
        pass

    def before_epoch(self, epoch):
        pass

    def each_batch(self, batch):
        if (batch+1)%10 == 0:
            self.saver.save(self.session, self.config.savedir)
            print('Batch: ', batch+1, flush=True)

    def after_epoch(self, epoch):
        self.saver.save(self.session, self.config.savedir)
        print('Epoch: ', epoch+1, flush=True)

    def after_train(self):
        self.saver.save(self.session, self.config.savedir)

    def test(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
        print('Session is closed')

    def get_feed_dict(self):
        result = {}
        for i in range(len(self.config.gpu.split(','))):
            values = self.ds.train.next_batch(self.config.batch_size)
            for tensor, value in zip(self.ts.sub_ts[i]['inputs'], values):
                result[tensor] = value
        return result


if __name__ == '__main__':
    print(os.sep)