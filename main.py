# main.py
#
import numpy as np
import bnn
#
# get the data from the datafile_ml100k
datafile = 'datafile_ml100k.npz'
with np.load(datafile) as f:
    Ratings = f['Ratings']  # 943 users x 1682 movies
    Users = f['Users']      # 943 users x 5 fields
    Items = f['Items']      # 1682 movies x 20 fields
#
# init vbls
batch_size = 10
#
#   set up the array of examples
num_users = 943
num_items = 1682
example_array = np.zeros([num_users * num_items, 25])
labels = np.zeros([num_user * num_items, 1])
for i in range(num_users):
    for j in range(num_items):
        # first 1682 rows of batch correspond to first user;
        #  second 1682 rows of batch correspond to second user, etc.
        example_array[i+j, 0:4] = Users[i, :]
        example_array[i+j, 5:24] = Items[j, :]
        labels[i+j, 0] = Ratings[i, j]
#
examples = np.array(example_array)
ground_truth = np.array(labels)
#
# examples now holds array of all examples; ground_truth holds array of all labels
# randomize the examples
p = np.random.permutation(len(ground_truth))
random_examples = examples[p]
random_labels = ground_truth[p]
#  set up the Bayesian network model
model = bnn.BNN(25, 1)
# grab a batch of num_ex examples, so that inputs is 10 x 25 (10 examples, 25 features per example)
# then train for each batch
num_ex = 10
inputs = []
targets = []
for i in range(num_ex):
    inputs.append(random_examples[i, :])
    targets.append(random_labels[i, :])
    model.train(inputs, targets)
#
# call the forward method of the Bayesian network model to do a fwd inference pass
#
inputs = random_examples[11, :]
output = model.forward(inputs)
print "output is: ", output
