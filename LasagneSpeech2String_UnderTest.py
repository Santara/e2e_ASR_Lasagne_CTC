import numpy as np
import theano
import theano.tensor as T
import lasagne
import pdb
from ctc import CTCLayer
import pickle
import progress_bar as pb 
# pdb.set_trace()
####### SETTING DEFAULT HYPER-PARAMETER VALUES #######

#Input representation size
INPUT_SIZE = 13

#Hidden layer hyper-parameters
N_HIDDEN = 100
HIDDEN_NONLINEARITY = 'rectify'

#Learning rate
LEARNING_RATE = 0.001

#Number of training sequences (here sentences or examples) in each batch
N_BATCH = 1

#Gradient clipping
GRAD_CLIP = 100

#How often we check the output
EPOCH_SIZE = 100

#Number of epochs to train the system on
NUM_EPOCHS = 1000

logspace = True
# def main(num_epochs = NUM_EPOCHS):
print("Building the network...")
given_labels = T.ivector('given_labels')
l_in = lasagne.layers.InputLayer(shape = (N_BATCH, None, INPUT_SIZE)) #One-hot represenntation of character indices
l_mask = lasagne.layers.InputLayer(shape = (None, None))

l_recurrent = lasagne.layers.RecurrentLayer(incoming = l_in, num_units=N_HIDDEN, mask_input = l_mask, learn_init=True)
RecurrentOutput = lasagne.layers.get_output(l_recurrent)

n_batch, n_time_steps, n_features = l_in.input_var.shape

# l_reshape = lasagne.layers.ReshapeLayer(l_recurrent, (-1, N_HIDDEN))

# l_reshape_out = lasagne.layers.get_output(l_reshape)

# l_dense = lasagne.layers.DenseLayer(l_reshape, num_units=INPUT_SIZE, nonlinearity = lasagne.nonlinearities.softmax)


# l_out = CTCLayer(lasagne.layers.get_output(l_dense).T, given_labels, INPUT_SIZE, logspace)
# l_out = lasagne.layers.ReshapeLayer(l_dense, (n_batch, n_time_steps, n_features))


#Training the network
# target_values = T.ivector('target_output') #A vector of character indices linearized over the sequences and batches
# mask = T.matrix('mask')

#Getting the expression for the output (we extract it from the softmax layer to make sure it is compatible with the cross entropy loss function of theano)
# network_output = lasagne.layers.get_output(l_dense)

# #Calculating the CTC-loss
# cost = l_out.cost
# cost = T.mean(lasagne.objectives.categorical_crossentropy(network_output, target_values))
# cost.reshape([n_batch, n_time_steps, n_features])

# lasagne_params = lasagne.layers.get_all_params(l_dense)
# CTC_params = l_out.params
# all_params = lasagne_params.extend(CTC_params)
# lasagne_gradient = theano.grad(cost, wrt=lasagne_params)
# CTC_gradient = theano.grad(cost, wrt=CTC_params)

# lasagne_updates = lasagne.updates.adam(lasagne_gradient, lasagne_params) 
# CTC_updates = lasagne.updates.adam(CTC_gradient, CTC_params) 
# updates = lasagne_updates.update(CTC_updates)

# train = theano.function(inputs=[l_in.input_var, given_labels, l_mask.input_var],	outputs=[cost],	updates=updates)

# compute_cost = theano.function(	inputs=[l_in.input_var, given_labels, l_mask.input_var], 	outputs=[cost], )


#################################################################################
with open('TIMIT_data_prepared_for_CTC.pkl','rb') as f:
	data = pickle.load(f)

x = data['x']
y = data['y_indices']
mask = data['mask']
pdb.set_trace()

inp = np.zeros([1, x.shape[1], x.shape[2]], dtype=theano.config.floatX)
inp[0] = x[0]	
msk = np.zeros([1,mask.shape[1]], dtype=theano.config.floatX)
msk[0] = mask[0]
tgt = y[0]	

# l_reshape_out_val = l_reshape_out.eval({l_in.input_var: inp, l_mask.input_var: msk})
RecurrentOutputValue = RecurrentOutput.eval({l_in.input_var: inp, l_mask.input_var: msk})
# SoftmaxOutputValue = network_output.eval({l_in.input_var: inp, l_mask.input_var: msk})

# print(
# 	"Input shape: {}"
# 	"\nMask shape: {}"
# 	"\nTarget length: {}"
# 	"\nReshape layer output size: {}"
# 	"\nOutput of the recurrent layer shape: {}"
# 	"\nOutput of the softmax layer shape: {}".format(inp.shape, msk.shape, len(tgt), l_reshape_out_val.shape, RecurrentOutputValue.shape, SoftmaxOutputValue.shape)
# 	)


# CostValue = cost.eval({l_in.input_var: inp, l_mask.input_var: msk, given_labels: tgt})


# cost_vector = []
# for epoch in range(NUM_EPOCHS):	
# 	print("\nEpoch: {}".format(epoch))
# 	shuffle_order = np.random.permutation(x.shape[0])
# 	x = x[shuffle_order, :]
# 	y = [y[i] for i in shuffle_order]
# 	# y_merged = y.reshape([-1])
# 	mask = mask[shuffle_order, :]
# 	# mask_merged = mask.reshape([-1])

# 	num_batches = x.shape[0]
# 	mean_cost = 0
# 	for batch in range(num_batches):
# 		pb.show_progress(float(batch/num_batches))
# 		inp = np.zeros([1, x.shape[1], x.shape[2]], dtype=theano.config.floatX)
# 		inp[0] = x[batch]	
# 		msk = np.zeros([1,mask.shape[1]], dtype=theano.config.floatX)
# 		msk[0] = mask[batch]
# 		tgt = y[batch]	
# 		cost = np.float64(train(inp,tgt,msk)[0])
# 		mean_cost += cost
# 	print("\ncost = {}".format(mean_cost/num_batches) )

# 	cost_vector.append(cost)

# output_file_name = 'CTC_cost_vector.pkl'
# with open(output_file_name,'wb') as f:
# 	pickle.dump(cost_vector,f,protocol=3)
