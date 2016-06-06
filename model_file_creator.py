import sys
import numpy as np

npz_model_file_name = sys.argv[1]
with np.load(npz_model_file_name) as in_file:
	param_values = [in_file['arr_%d' % i] for i in range(len(in_file.files))]


f = open('model_CTC.txt','w')

#Number_of_layers
print >> f, '2'

#'#1:', 'RecurrentLayer', 'U'
print >> f, 'RU'

#'W_in_to_hid'
print >> f,  '10', '39'

print >> f, str(param_values[0].reshape(-1))[1:-1]

#'W_hid_to_hid'
print >> f,  '10', '10'

print >> f, str(param_values[2].reshape(-1))[1:-1]

#'b'
print >> f,  '10'

print >> f, str(param_values[1])[1:-1]

#'#2:', 'DenseLayer', 'U'
print >> f, 'D'

#'W',
print >> f,  '30', '10'

print >> f, str(param_values[3].reshape(-1))[1:-1]

#'b',
print >> f,  '30'

print >> f, str(param_values[4])[1:-1]