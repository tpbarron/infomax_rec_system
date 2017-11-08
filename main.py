# main.py
#
import torch
from torch.autograd import Variable
import numpy as np
import bnn
from sklearn.preprocessing import MinMaxScaler

# get the data from the datafile_ml100k
datafile = 'datafile_ml100k.npz'
with np.load(datafile) as f:
    Ratings = f['Ratings']  # 943 users x 1682 movies
    Users = f['Users']      # 943 users x 5 fields
    Items = f['Items']      # 1682 movies x 20 fields

#   set up the array of examples
num_users = 943
num_items = 1682
example_array = np.zeros([num_users * num_items, 25])
labels = np.zeros([num_users * num_items, 1])
for i in range(num_users):
    for j in range(num_items):
        # first 1682 rows of batch correspond to first user;
        #  second 1682 rows of batch correspond to second user, etc.
        example_array[i+j, 0:5] = Users[i, :]
        example_array[i+j, 5:25] = Items[j, :]
        labels[i+j, 0] = Ratings[i, j]

# examples & targets
examples = np.array(example_array)[0:1682,:]
ground_truth = np.array(labels)[0:1682]
nz = np.nonzero(ground_truth)[0]
examples = examples[nz,:]
ground_truth = ground_truth[nz]
ground_truth_one_hot = np.zeros((len(ground_truth), 3))
for i in range(len(ground_truth)):
    if ground_truth[i] == -1:
        ground_truth_one_hot[i][0] = 1
    elif ground_truth[i] == 0.01:
        ground_truth_one_hot[i][1] = 1
    elif ground_truth[i] == 1:
        ground_truth_one_hot[i][2] = 1

print ("Samples: ", examples.shape, ground_truth.shape)

scaler = MinMaxScaler()
scaler.fit(examples)
examples = scaler.transform(examples)
print (examples.min(), examples.max())

# shuffle the data
# examples now holds array of all examples; ground_truth holds array of all labels
# randomize the examples
p = np.random.permutation(len(ground_truth_one_hot))
random_examples = examples[p]
random_labels = ground_truth_one_hot[p]

#  set up the Bayesian network model
model = bnn.BNN(25, 3)
# grab a batch of num_ex examples, so that inputs is 10 x 25 (10 examples, 25 features per example)
# then train for each batch

print (random_examples.shape)
epochs = 500
batch_size = 32
num_batches = int(np.ceil(len(random_examples) / batch_size))

def compute_error():
    error = 0.0
    for i in range(len(random_labels)):
        inputs = random_examples[i, :]
        true_out = random_labels[i, :]
        inputs = inputs.reshape((1, len(inputs)))
        inputs = Variable(torch.from_numpy(inputs), volatile=True).float()
        output = model.forward(inputs)
        error += ((true_out - output[0].data.numpy())**2.).sum() / len(true_out)
    print ("Mean Error:", error / len(random_labels))


e = 0
compute_error()
while e < epochs:
    print ("Epoch: ", e)
    b = 0
    while b < num_batches:
        # if b % 100 == 0:
        # print ("Batch: ", b, " of ", num_batches)

        ind_start = b*batch_size
        ind_end = (b+1)*batch_size

        inputs = random_examples[ind_start:ind_end,:]
        targets = random_labels[ind_start:ind_end,:]
        loss = model.train(inputs, targets)
        b += 1
    compute_error()
    e += 1


#
# call the forward method of the Bayesian network model to do a fwd inference pass
#

for i in range(len(random_labels)):
    inputs = random_examples[i, :]
    true_out = random_labels[i, :]
    inputs = inputs.reshape((1, len(inputs)))
    inputs = Variable(torch.from_numpy(inputs), volatile=True).float()
    output = model.forward(inputs)
    print ("Est/true:", output[0].data.numpy(), true_out)
