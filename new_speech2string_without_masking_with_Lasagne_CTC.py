
# coding: utf-8

# Imports

# In[82]:

import pickle as pkl
import numpy as np
import theano
import theano.tensor as T
import lasagne 
from lasagne.layers import InputLayer, DenseLayer, RecurrentLayer, NonlinearityLayer, ReshapeLayer, get_output, get_all_params
import ctc_cost


# load data

# In[83]:

f = open('TIMIT_data_prepared_for_CTC.pkl','rb')
data = pkl.load(f)
inp = data['x'].astype(theano.config.floatX)
msk = data['mask']
tgt = data['y_indices']


# build the network

# In[84]:

input_size = 13
hidden_size = 10
num_output_classes = 26
output_size = num_output_classes+1
batch_size = None
input_seq_length = None


# In[85]:

# x = T.dtensor3('inputs') #Change this to ftensor3 when floatX=float32
y = T.imatrix('targets')


# In[86]:

l_in = InputLayer(shape=(batch_size, input_seq_length, input_size))
n_batch, n_time_steps, n_features = l_in.input_var.shape #Unnecessary in this version. Just collecting the info so that we can reshape the output back to the original shape
l_rec = RecurrentLayer(l_in, num_units=hidden_size)
l_rec_reshaped = ReshapeLayer(l_rec, (-1,hidden_size))
l_out = DenseLayer(l_rec_reshaped, num_units=output_size)
l_out_reshaped = ReshapeLayer(l_out, (n_batch, n_time_steps, output_size))#Reshaping back
l_out_softmax = NonlinearityLayer(l_out, nonlinearity=lasagne.nonlinearities.softmax)
l_out_softmax_reshaped = ReshapeLayer(l_out_softmax, (n_batch, n_time_steps, output_size))


# In[87]:

output_logits = get_output(l_out_reshaped)
output_softmax = get_output(l_out_softmax_reshaped)


# In[88]:

all_params = get_all_params(l_out,trainable=True)
print all_params


# In[89]:

pseudo_cost = ctc_cost.pseudo_cost(y, output_logits)


# In[90]:

pseudo_cost_grad = T.grad(pseudo_cost.sum() / n_batch, all_params)


# In[91]:

#Disputed area, does not compile >_<
true_cost = ctc_cost.cost(y, output_softmax)
cost = T.mean(true_cost)


# In[92]:

# theano.printing.pydotprint(output_logits, outfile="./compute_graph.png", var_with_name_simple=True)  
# theano.printing.debugprint(output_logits) 


# In[93]:

shared_learning_rate = theano.shared(lasagne.utils.floatX(0.01))
updates = lasagne.updates.rmsprop(pseudo_cost_grad, all_params, learning_rate=shared_learning_rate)


# In[96]:

theano.config.exception_verbosity='high'
train = theano.function([l_in.input_var,y], [output_logits, output_softmax, cost, pseudo_cost], updates=updates)
# train = theano.function([l_in.input_var,y], [output_logits, output_softmax, cost, pseudo_cost], updates=updates)


# In[115]:

inp0 = inp[0].astype(theano.config.floatX)
msk0 = msk[0].astype(np.bool)
inp0 = inp0[msk0]
inp00= np.asarray([inp0])
tgt0 = np.asarray(tgt[0],dtype=np.int16)
tgt00 = np.asarray([tgt0])
# inp00 = np.asarray([inp0[msk0]])
print inp00.shape, msk0.shape, tgt00.shape


# In[156]:

a,b,c,d = train(inp00,tgt00)
print c

