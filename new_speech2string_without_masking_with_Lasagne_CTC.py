
# coding: utf-8

# Imports
# ======

# In[1]:

import pickle as pkl
import numpy as np
import theano
import theano.tensor as T
import lasagne 
from lasagne.layers import InputLayer, DenseLayer, RecurrentLayer, NonlinearityLayer, ReshapeLayer, get_output, get_all_params, get_all_param_values, ElemwiseSumLayer
import ctc_cost
from time import time
from TIMIT_utils import index2char_TIMIT
from special_activations import clipped_relu
import sys


# load data
# ====

# In[2]:

f = open('TIMIT_data_prepared_for_CTC.pkl','rb')
data = pkl.load(f)
inp = data['x']
inp1 = data['inputs']
msk = data['mask']
tgt = data['y_indices']
char = data['chars']


# build the network
# ===

# In[3]:

input_size = len(inp1[0][0])
hidden_size = 1824
num_output_classes = len(char)
learning_rate = 0.0627
output_size = num_output_classes+1
batch_size = None
input_seq_length = None
gradient_clipping = 5


# In[5]:

# x = T.dtensor3('inputs') #Change this to ftensor3 when floatX=float32
y = T.imatrix('targets')


# In[6]:

l_in = InputLayer(shape=(batch_size, input_seq_length, input_size))
n_batch, n_time_steps, n_features = l_in.input_var.shape #Unnecessary in this version. Just collecting the info so that we can reshape the output back to the original shape
h_1 = DenseLayer(l_in, num_units=hidden_size, nonlinearity=clipped_relu)
l_rec_forward = RecurrentLayer(h_1, num_units=hidden_size, grad_clipping=gradient_clipping, nonlinearity=clipped_relu)
l_rec_backward = RecurrentLayer(h_1, num_units=hidden_size, grad_clipping=gradient_clipping, nonlinearity=clipped_relu, backwards=True)
l_rec_accumulation = ElemwiseSumLayer([l_rec_forward,l_rec_backward])
l_rec_reshaped = ReshapeLayer(l_rec_accumulation, (-1,hidden_size))
l_out = DenseLayer(l_rec_reshaped, num_units=output_size, nonlinearity=lasagne.nonlinearities.linear)
l_out_reshaped = ReshapeLayer(l_out, (n_batch, n_time_steps, output_size))#Reshaping back
l_out_softmax = NonlinearityLayer(l_out, nonlinearity=lasagne.nonlinearities.softmax)
l_out_softmax_reshaped = ReshapeLayer(l_out_softmax, (n_batch, n_time_steps, output_size))


# In[7]:

output_logits = get_output(l_out_reshaped)
output_softmax = get_output(l_out_softmax_reshaped)


# In[8]:

all_params = get_all_params(l_out,trainable=True)
# print all_params==[l_rec.W_in_to_hid, l_rec.b, l_rec.W_hid_to_hid, l_out.W, l_out.b]


# In[9]:

pseudo_cost = ctc_cost.pseudo_cost(y, output_logits)


# In[10]:

pseudo_cost_grad = T.grad(pseudo_cost.sum() / n_batch, all_params)


# In[11]:

#Disputed area, does not compile >_<
true_cost = ctc_cost.cost(y, output_softmax)
cost = T.mean(true_cost)


# In[12]:

# theano.printing.pydotprint(output_logits, outfile="./compute_graph.png", var_with_name_simple=True)  
# theano.printing.debugprint(output_logits) 


# In[13]:


shared_learning_rate = theano.shared(lasagne.utils.floatX(learning_rate))
updates = lasagne.updates.rmsprop(pseudo_cost_grad, all_params, learning_rate=shared_learning_rate)


# In[14]:

theano.config.exception_verbosity='high'
train = theano.function([l_in.input_var,y], [output_logits, output_softmax, cost, pseudo_cost], updates=updates)
# train = theano.function([l_in.input_var,y], [output_logits, output_softmax, cost, pseudo_cost], updates=updates)


# In[15]:

# inp0 = inp1[0]#.astype(theano.config.floatX)
# # msk0 = msk[0].astype(np.bool)
# # inp0 = inp0[msk0]
# inp00= np.asarray([inp0],dtype=theano.config.floatX)
# tgt0 = np.asarray(tgt[0],dtype=np.int16)
# tgt00 = np.asarray([tgt0])
# # inp00 = np.asarray([inp0[msk0]])
# # print inp00.shape, msk0.shape, tgt00.shape
# print inp00.shape, tgt00.shape


# In[16]:

# # a,b,c,d = train(inp00,tgt00)
# # print c
# # y_ = output_softmax.eval({l_in.input_var:inp00})
# for elem in pseudo_cost_grad:
#     print elem.eval({l_in.input_var:inp00, y:tgt00})
    


# In[ ]:

# # y_.shape
# print np.argmax(y_,axis=2)
# from TIMIT_utils import index2char_TIMIT
# print index2char_TIMIT(np.argmax(y_, axis=2)[0])
# print index2char_TIMIT(tgt0)


# In[ ]:

num_epochs = 1000
num_training_samples = len(inp1)
for epoch in range(num_epochs):
    t = time()
    cost = 0
    failures = []
    for i in range(num_training_samples):
        curr_inp = inp1[i]
#         curr_msk = msk[i].astype(np.bool)
#         curr_inp = curr_inp[curr_msk]
        curr_inp = np.asarray([curr_inp],dtype=theano.config.floatX)
        curr_tgt = np.asarray(tgt[i],dtype=np.int16)
        curr_tgt = np.asarray([curr_tgt])
        try:
            _,_,c,_=train(curr_inp,curr_tgt)
            cost += c
        except IndexError:
            failures.append(i)
            print 'Current input seq: ', curr_inp
            print 'Current output seq: ', curr_tgt
            sys.exit(IndexError)
    print 'Epoch: ', epoch, 'Cost: ', float(cost/(num_training_samples-len(failures))), ', time taken =', time()-t
#     print 'Exceptions: ', len(failures), 'Total examples: ', num_training_samples
    if epoch%10==0:        
        #Save the model
        np.savez('CTC_model.npz', *get_all_param_values(l_out_softmax_reshaped, trainable=True))
        for i in range(2):
            curr_inp = inp1[i]
            curr_inp = np.asarray([curr_inp],dtype=theano.config.floatX)
            curr_tgt = np.asarray(tgt[i],dtype=np.int16)
            curr_out = output_softmax.eval({l_in.input_var:curr_inp})
            print 'Predicted:', index2char_TIMIT(np.argmax(curr_out, axis=2)[0])
            print 'Target:', index2char_TIMIT(curr_tgt)


# In[ ]:

# num_epochs = 100
# num_training_samples = len(inp)
# for epoch in range(num_epochs):
#     t = time()
#     cost = 0
#     for i in range(num_training_samples):
#         curr_inp = inp[i].astype(theano.config.floatX)
#         curr_msk = msk[i].astype(np.bool)
#         curr_inp = curr_inp[curr_msk]
#         curr_inp = np.asarray([curr_inp])
#         curr_tgt = np.asarray(tgt[i],dtype=np.int16)
#         curr_tgt = np.asarray([curr_tgt])
#         _,_,c,_=train(curr_inp,curr_tgt)
#         cost += c
#     print float(cost/num_training_samples), ', time taken =', time()-t


# In[ ]:

print input_size

