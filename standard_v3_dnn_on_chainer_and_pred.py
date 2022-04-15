

#!/usr/bin/env python3
# LANL Earthquake DNN Approach (onchainer)
import pandas as pd
import numpy as np
import cupy as xp
import scipy
import time
import pickle 
import chainer
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import chainer.optimizers as O
from chainer import reporter
from chainer import training
from chainer.training import extensions
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import math
gpu = 0
class WindowIterator(chainer.dataset.Iterator):
    """Dataset iterator to create a batch of sequences at different positions.

    This iterator returns a pair of the current words and the context words.
    """

    def __init__(self, x =None,static_x =None,static_step_X =None,y =None,
                batch_size=None, 
                repeat=True): 
        self.x = x
        self.static_x = static_x
        self.static_step_X = static_step_X
        self.y = y 
        self.batch_size = batch_size
        self._repeat = repeat
        # order is the array which is shuffled ``[window, window + 1, ...,
        # len(dataset) - window - 1]``
        
        self.current_position = 0
        # Number of completed sweeps over the dataset.  In this case, it is
        # incremented if every word is visited at least once after the last
        # increment.
        self.epoch = 0
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False

    def __next__(self):
        """This iterator returns a list representing a mini-batch.

        Each item indicates a different position in the original sequence.
        """
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        mini_x = self.x[i:i_end]
        mini_static_x = self.static_x[i:i_end]
        mini_static_step_X = [x[i:i_end] for x in self.static_step_X]
        mini_y = self.y[i:i_end] 
        
        if i_end >= len(self.x):
            self.epoch += 1
            self.is_new_epoch = True
            self.current_position = 0
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return  mini_x,mini_static_x,mini_static_step_X,mini_y

    @property
    def epoch_detail(self):
        return self.epoch + float(self.current_position) / len(self.x)

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)
def convert(batch, device):
    mini_x,mini_static_x,mini_static_step_X,mini_y = batch
    if device >= 0:
        mini_x = cuda.to_gpu(mini_x)
        mini_static_x = cuda.to_gpu(mini_static_x)
        mini_static_step_X = cuda.to_gpu(mini_static_step_X)
        mini_y = cuda.to_gpu(mini_y)
    return mini_x,mini_static_x,mini_static_step_X,mini_y

class standar_model(Chain):
    """Definition of Skip-gram Model""" 
    def __init__(self):
        super(standar_model, self).__init__()

        with self.init_scope():
            self.L1 = L.Linear(200, 1500)
            self.bn = L.BatchNormalization(1500)

            self.L_stat1 = L.Linear(16, 1500)
            self.bn_stat1 = L.BatchNormalization(1500)
            for i in range(20):  
                self.add_link('linear_step_x' + str(i),L.Linear(16,500))
                self.add_link('linear_bn_x' + str(i), L.BatchNormalization(500))
            self.L2 = L.Linear(None,10000)
            self.L3 = L.Linear(10000, 5000) 
            self.L4 = L.Linear(5000, 2) 
    def forward(self, mini_x,mini_static_x,mini_static_step_X,mini_y):
        mini_x = xp.array(mini_x,dtype=xp.float32)
        mini_static_x = xp.array(mini_static_x,dtype=xp.float32)
        mini_y = xp.array(mini_y,dtype=xp.int32)

        h1 = self.L1(mini_x)
        bn1 = self.bn(h1)

        h1_stat1 = self.L_stat1(mini_static_x)
        bn_stat1 = self.bn_stat1(h1_stat1)
        wk_h1 = Variable()
        for i,item in enumerate(mini_static_step_X):
            if i == 0:
                item = xp.array(item,dtype=xp.float32)
                h3 = self['linear_step_x' + str(i)](item)
                wk_h1 = self['linear_bn_x' + str(i)](h3)
            else:
                item = xp.array(item,dtype=xp.float32)
                LL1 = self['linear_step_x' + str(i)](item)
                LL2 = self['linear_bn_x' + str(i)](LL1)
                wk_h1= F.concat((h3, LL2), axis=1)

        con1 = F.concat((bn1, bn_stat1), axis=1)
        con1 = F.concat((con1, wk_h1), axis=1)
        h2 = F.sigmoid(self.L2(con1))
        h3 = self.L3(h2)
        y = self.L4(h3)
        loss = F.softmax_cross_entropy(y, mini_y) 
        acc = F.accuracy(y, mini_y)
        report({'accuracy': acc.data}, self)
        report({'loss': loss.data}, self)
        return loss  
    def predict(self, mini_x,mini_static_x,mini_static_step_X):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            mini_x = xp.array(mini_x,dtype=xp.float32)
            mini_static_x = xp.array(mini_static_x,dtype=xp.float32)

            h1 = self.L1(mini_x)
            bn1 = self.bn(h1)

            h1_stat1 = self.L_stat1(mini_static_x)
            bn_stat1 = self.bn_stat1(h1_stat1)
            wk_h1 = Variable()
            for i,item in enumerate(mini_static_step_X):
                if i == 0:
                    item = xp.array(item,dtype=xp.float32)
                    h3 = self['linear_step_x' + str(i)](item)
                    wk_h1 = self['linear_bn_x' + str(i)](h3)
                else:
                    item = xp.array(item,dtype=xp.float32)
                    LL1 = self['linear_step_x' + str(i)](item)
                    LL1 = self['linear_bn_x' + str(i)](LL1)
                    wk_h1= F.concat((h3, LL1), axis=1)

            con1 = F.concat((bn1, bn_stat1), axis=1)
            con1 = F.concat((con1, wk_h1), axis=1)
            h2 = F.sigmoid(self.L2(con1))
            h3 = self.L3(h2)
            y = self.L4(h3)
        
        return y


def summary(arr):
    list1 = [np.amin(arr), np.amax(arr),  np.mean(arr), np.median(arr),
            np.percentile(arr,10),np.percentile(arr,20),np.percentile(arr,30),np.percentile(arr,40),np.percentile(arr,50),np.percentile(arr,60),np.percentile(arr,70),
            np.percentile(arr,80),np.percentile(arr,90),np.var(arr),np.std(arr),np.average(arr)]
    list1 = [item if  item is not None else 0 for item in list1]
    return list1
def create_train_data(proc_data):
    x = []
    y = []
    static_x = []
    statics_step10 =[]
    train = []
    ind0 =0
    ind1 =0
    train = pd.DataFrame(proc_data, columns=["var_" + str(i) for i in range(200)])
    for i1 in range(21)[1:]:
        ind1 = i1 * 10
        statics_step10.append(pd.DataFrame(proc_data, columns=["var_" + str(i) for i in range(200)[ind0:ind1]]))
        ind0 = ind1
    x = np.array(train.values.tolist(),dtype=np.float32)
    static_x = np.array([summary(x_value) for x_value in x],dtype=np.float32)
    static_step_X = np.array([[summary(x_value1) for x_value1 in np.array(x1.values.tolist())] for x1 in statics_step10],dtype=np.float32) 
    y =np.array( np.round(proc_data['target']),dtype=np.int32) 
    return  x ,static_x,static_step_X,y
def create_pred_data(proc_data):
    x = []
    y = []
    static_x = []
    statics_step10 =[]
    train = []
    ind0 =0
    ind1 =0
    id = proc_data['ID_code'].values.tolist()
    train = pd.DataFrame(proc_data, columns=["var_" + str(i) for i in range(200)])
    for i1 in range(21)[1:]:
        ind1 = i1 * 10
        statics_step10.append(pd.DataFrame(proc_data, columns=["var_" + str(i) for i in range(200)[ind0:ind1]]))
        ind0 = ind1
    x = np.array(train.values.tolist(),dtype=np.float32)
    static_x = np.array([summary(x_value) for x_value in x],dtype=np.float32)
    static_step_X = np.array([[summary(x_value1) for x_value1 in np.array(x1.values.tolist())] for x1 in statics_step10],dtype=np.float32) 
    return id, x ,static_x,static_step_X
if __name__ == '__main__':
    print("start training")
    proc_data1 = pd.read_csv(r"../input/train.csv") 
    x ,static_x,static_step_X ,y = create_train_data(proc_data1) 
    print(len(x))
    model = standar_model()
    model.compute_accuracy = True
    dir1 = 'standar/'
    project_name = 'standar_'
    #if gpu >= 1:
    model.to_gpu(gpu)
    optimizer = optimizers.Adam() 
    optimizer.setup(model)
    # Setup optimizer 
    train_iter = WindowIterator(x =x,static_x=static_x,static_step_X=static_step_X ,y=y, batch_size=1024)
    
    updater = training.updaters.StandardUpdater(train_iter, optimizer, converter=convert, device=gpu)
    trainer = training.Trainer(updater, (20, 'epoch'), out=dir1 + "result")
    trainer.extend(extensions.LogReport(log_name= project_name + 'log.txt'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar()) 
   

    # trainer.extend(extensions.snapshot(filename=project_name +
    # 'snapshot_epoch-{.updater.epoch}'))
    # trainer.extend(extensions.snapshot_object(model.predictor,
    # filename='model_epoch-{.updater.epoch}'))
    trainer.run()

    model.to_cpu()
    serializers.save_npz('learned.model', model)
    serializers.save_npz('learned.state', optimizer)
    
    proc_data = pd.read_csv(r"../input/test.csv") 
    #proc_data = pd.read_csv(r"C:\Users\ya\Downloads\santander-customer-transaction-prediction\test.csv",nrows=100) 
    id, x ,static_x,static_step_X = create_pred_data(proc_data) 
    dir1 = 'standar/'
    project_name = 'standar_'
    if gpu != -1:
        model.to_gpu(gpu) 
    # Setup optimizer   
    result = []
    result.append(["ID_code","target"])

    pred_source = []
    batchsize = 100
    ind1 = 0
    ind2 = 0
    cont_int = -1
    for k in range(1, math.ceil(len(id) / batchsize) + 1):
        print(str(ind1))
        ind2 = batchsize * k
        if ind2 > len(id):
            ind2 = len(id)
        mini_x = x[ind1:ind2]
        mini_static_x =static_x[ind1:ind2]
        mini_static_step_X = [x[ind1:ind2] for x in static_step_X] 
        pred =  F.softmax(model.predict(mini_x,mini_static_x,mini_static_step_X )).data.tolist()
        for i,item in enumerate(pred):  
            result.append([id[ind1+ i],str("%.10f" % item[1] )])
        ind1 = ind2
        
        
    import csv 
    with open('submission_hukuzatsu.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
        writer.writerows(result) # 2次元配列も書き込める
        