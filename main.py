# main.py
#
import torch
from torch.autograd import Variable
import numpy as np
import bnn
from sklearn.preprocessing import MinMaxScaler
import simple_model

# get the data from the datafile_ml100k
datafile = 'datafile_ml100k.npz'
with np.load(datafile) as f:
    Ratings = f['Ratings']  # 943 users x 1682 movies
    Users = f['Users']      # 943 users x 5 fields
    Items = f['Items']      # 1682 movies x 20 fields
# debug
print("first row of Ratings: ", Ratings[0, :])

#   set up the array of examples
num_users = 943
num_items = 1682
example_array = np.zeros([num_users * num_items, 25])
labels = np.zeros([num_users * num_items, 1])
index = 0
for i in range(num_users):
    for j in range(num_items):
        # first 1682 rows of batch correspond to first user;
        #  second 1682 rows of batch correspond to second user, etc.
        example_array[index, 0:5] = Users[i, :]
        example_array[index, 5:25] = Items[j, :]
        labels[index, 0] = Ratings[i, j]
        index += 1

# debug
print("first row of labels: ", labels[0:1682, 0])
# examples & targets
examples = np.array(example_array)[0:1682,:]
ground_truth = np.array(labels)[0:1682]
nz = np.nonzero(ground_truth)[0]
examples = examples[nz,:]
ground_truth = ground_truth[nz]
# two-way ratings
ground_truth_one_hot = np.zeros((len(ground_truth), 2))
for i in range(len(ground_truth)):
    if ground_truth[i] == 0.01:
        ground_truth_one_hot[i][0] = 1      # [1 0] means rating was 1-3
    elif ground_truth[i] == 1:
        ground_truth_one_hot[i][1] = 1      # [0 1] means rating was 4-5

"""
# three-way ratings
ground_truth_one_hot = np.zeros((len(ground_truth), 3))
for i in range(len(ground_truth)):
    if ground_truth[i] == :
        ground_truth_one_hot[i][0] = 1
    elif ground_truth[i] == 0.01:
        ground_truth_one_hot[i][1] = 1
    elif ground_truth[i] == 1:
        ground_truth_one_hot[i][2] = 1
"""

print ("Samples: ", examples.shape, ground_truth_one_hot.shape)
print ("examples: ", examples)
print ("labels: ", ground_truth_one_hot)

scaler = MinMaxScaler()
scaler.fit(examples)
examples = scaler.transform(examples)
print (examples.min(), examples.max())

# shuffle the data
# examples now holds array of all examples; ground_truth holds array of all labels
# randomize the examples
p = np.random.permutation(len(ground_truth_one_hot[:, 0]))
random_examples = examples[p]
random_labels = ground_truth_one_hot[p]

#  set up the model
#  model_type:  BNN for Bayesian network, FC for fully-connected/dense/linear model
model_type = 'FC'
if model_type == 'BNN':
    model = bnn.BNN(25, 2, lr=0.001)
elif model_type == 'FC':
    model = simple_model.FC(25, 2)

# grab a batch of num_ex examples, so that inputs is 10 x 25 (10 examples, 25 features per example)
# then train for each batch

print (random_examples.shape)
epochs = 500
batch_size = 32
# num_batches = int(np.ceil(len(random_examples) / batch_size))
num_batches = int(np.floor(len(random_examples) / batch_size))

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


# save the Model
torch.save(model.state_dict(), './model' + '-' + model_type + '.pth')
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
