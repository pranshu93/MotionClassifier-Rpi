from __future__ import print_function
import csv
import os
import numpy as np
import sys
import tensorflow as tf
from tensorflow.contrib import rnn

def forward_iter(data, labels, index, code):
    batchx = data[index];  batchy = labels[index];
    if(code): sess.run(train_op, feed_dict={X: batchx.reshape((-1,timesteps, num_input)), Y:batchy})
    else: return(sess.run([accuracy, prediction], feed_dict={X: batchx.reshape((-1,timesteps, num_input)), Y: batchy}))

#file = open("mod_data.csv")
file = open(sys.argv[1])
reader = csv.reader(file)
data = []
ctr = 0
for row in reader:
    if(ctr): data.append(row[1:])
    ctr += 1
data = np.asarray(data, dtype=np.float32)
labels = data[:,-1]
data = data[:,:-1]

mean=np.mean(data,0)
std=np.std(data,0)
std[std[:]<0.00001]=1

ex_un_data = data.reshape(1,-1)
data=(data-mean)/std

ex_data = data.reshape(1,-1)
dpdncy = int(sys.argv[2])
cr_data = np.zeros((data.shape[0]-dpdncy,(dpdncy+1)*data.shape[1]))
for i in range(cr_data.shape[0]):
    cr_data[i] = ex_data[0,slice(i*data.shape[1],i*data.shape[1]+(dpdncy+1)*data.shape[1])]
cr_labels = labels[dpdncy:]

tf.set_random_seed(42)
np.random.seed(42)

learning_rate = 0.01
num_epochs = 100
batch_size = 25

num_input = data.shape[1] 
timesteps = dpdncy + 1
num_hidden = int(sys.argv[3])
num_classes = int(np.max(labels) - np.min(labels) + 1)

cr_labels=np.zeros((data.__len__(),num_classes)); cr_labels[np.arange(data.__len__()),np.array(labels.tolist(),dtype=int)]=1;

X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

weights = {'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))}
biases = {'out': tf.Variable(tf.random_normal([num_classes]))}

def RNN(x, weights, biases):
    x = tf.unstack(x, timesteps, 1)
    #lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1)
    lstm_cell = rnn.BasicRNNCell(num_hidden)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out'], outputs

logits,outs = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_variables(tf.local_variables()))) 

saver = tf.train.Saver()

init = tf.global_variables_initializer()

max_acc = 0

for i in range(num_epochs):
    num_iter = int(cr_data.__len__()/batch_size)
    [forward_iter(cr_data,cr_labels,slice(j*batch_size,(j+1)*batch_size),True) for j in range(num_iter)]
    forward_iter(cr_data,cr_labels,slice(num_iter*batch_size,cr_data.__len__()),True)
    acc,_ = forward_iter(cr_data,cr_labels,slice(0,cr_data.__len__()),False)
    print(acc)
    #if(max_acc < acc):
    #    max_acc = acc
        #saver.save(sess, modelloc + "/bestmodel.ckpt")	

#acc,hdot = forward_iter(cr_data,cr_labels,slice(0,1),False)
variables_names = [v.name for v in tf.trainable_variables()]
values = sess.run(variables_names)

for k, v in zip(variables_names, values):     
    if(k.find("Variable:0") != -1):
        FC_Weight = v
    if(k.find("Variable_1:0") != -1):
        FC_Bias = v    
    if(k.find("kernel") != -1):
        W = v[:num_input,:]; U = v[num_input:,:]
    if(k.find("bias") != -1):
        b = v
    #print ("Variable: ", k)
    #print (v)

def predict_fp():
    pred_lbls = []
    for i in range(cr_data.shape[0]):
        h = np.array(np.zeros((num_hidden,1)),dtype=float)
        for t in range(dpdncy + 1):
            #x = np.array(ex_data[0][slice(num_input*(i+t),num_input*(i+t+1))],dtype=float).reshape((-1,1))
            x = np.array((ex_un_data[0][slice(num_input*(i+t),num_input*(i+t+1))]-mean)/std,dtype=float).reshape((-1,1))
            h = np.array(np.matmul(np.transpose(W),x) + np.matmul(np.transpose(U),h) + b.reshape(-1,1) ,dtype=float)
            h = np.tanh(h)
        
        #print(np.matmul(np.transpose(h),FC_Weight) + FC_Bias)
        pred_lbls.append(np.argmax(np.matmul(np.transpose(h),FC_Weight) + FC_Bias))
    pred_lbls = np.array(pred_lbls)
    #print(labels[dpdncy:])
    _,pr = forward_iter(cr_data,cr_labels,slice(0,cr_data.__len__()),False)
    #print(np.argmax(pr, 1)) 
    #print(pred_lbls)
    #print(float((pred_lbls==np.argmax(pr, 1)).sum())/np.argmax(pr, 1).shape[0])

if(not(os.path.isdir("Parameters"))):
	os.system("mkdir Parameters")

np.save("Parameters/FC_Weight.npy",FC_Weight)
np.save("Parameters/FC_Bias.npy",FC_Bias)
np.save("Parameters/W.npy",W)
np.save("Parameters/U.npy",U)
np.save("Parameters/b.npy",b)
np.save("Parameters/mean.npy",mean)
np.save("Parameters/std.npy",std)
np.save("Parameters/num_hidden.npy",num_hidden)
np.save("Parameters/dpdncy.npy",dpdncy)

#print(FC_Weight)
#print(FC_Bias)
#print(W)
#print(U)
#print(b)

#predict_fp()
