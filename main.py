# main.py
#
import torch
from torch.autograd import Variable
import numpy as np
import bnn

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
examples = np.array(example_array)
ground_truth = np.array(labels)

# shuffle the data
# examples now holds array of all examples; ground_truth holds array of all labels
# randomize the examples
p = np.random.permutation(len(ground_truth))
random_examples = examples[p]
random_labels = ground_truth[p]

#  set up the Bayesian network model
model = bnn.BNN(25, 1)
# grab a batch of num_ex examples, so that inputs is 10 x 25 (10 examples, 25 features per example)
# then train for each batch

print (random_examples.shape)
epochs = 1
batch_size = 64
num_batches = int(np.ceil(len(random_examples) / batch_size))

e = 0
while e < epochs:
    print ("Epoch: ", e)
    b = 0
    while b < num_batches:
        if b % 100 == 0:
            print ("Batch: ", b, " of ", num_batches)

        ind_start = b*batch_size
        ind_end = (b+1)*batch_size

        inputs = random_examples[ind_start:ind_end,:]
        targets = random_labels[ind_start:ind_end,:]
        loss = model.train(inputs, targets)
        b += 1

    e += 1


#
# call the forward method of the Bayesian network model to do a fwd inference pass
#
inputs = random_examples[11, :]
inputs = inputs.reshape((1, len(inputs)))
inputs = Variable(torch.from_numpy(inputs), volatile=True).float()
output = model.forward(inputs)
print ("output is: ", output)
