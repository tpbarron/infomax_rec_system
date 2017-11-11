import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.autograd import Variable

import bnn
import simple_model

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='datafile_ml100k.npz', help='data file')
parser.add_argument('--model-type', type=str, default='BNN', help='model type, FC or BNN')
parser.add_argument('--batch-size', type=int, default=128, help='training batch size')
parser.add_argument('--epochs', type=int, default=10000, help='training epochs')
parser.add_argument('--load-model', type=str, default='', help='which model to load')
args = parser.parse_args()

# get the data from the datafile_ml100k
datafile = args.data_path

def load_data():
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

    # normalize all data
    scaler = MinMaxScaler()
    scaler.fit(example_array)
    example_array = scaler.transform(example_array)
    print (example_array.min(), example_array.max())

    movies = example_array[0:1682, 5:25]

    # debug
    print("first row of labels: ", labels[0:1682, 0])
    # examples & targets
    examples = example_array[0:1682,:]
    ground_truth = labels[0:1682]
    nz = np.nonzero(ground_truth)[0]
    examples = examples[nz,:]
    ground_truth = ground_truth[nz]

    ground_truth_one_hot = np.zeros((len(ground_truth), 1))
    for i in range(len(ground_truth)):
        if ground_truth[i] == 0.01:
            ground_truth_one_hot[i] = 0      # [0] means rating was 1-3
        elif ground_truth[i] == 1:
            ground_truth_one_hot[i] = 1      # [1] means rating was 4-5

    # two-way ratings
    # ground_truth_one_hot = np.zeros((len(ground_truth), 2))
    # for i in range(len(ground_truth)):
    #     if ground_truth[i] == 0.01:
    #         ground_truth_one_hot[i][0] = 1      # [1 0] means rating was 1-3
    #     elif ground_truth[i] == 1:
    #         ground_truth_one_hot[i][1] = 1      # [0 1] means rating was 4-5
    # three-way ratings
    # ground_truth_one_hot = np.zeros((len(ground_truth), 3))
    # for i in range(len(ground_truth)):
    #     if ground_truth[i] == :
    #         ground_truth_one_hot[i][0] = 1
    #     elif ground_truth[i] == 0.01:
    #         ground_truth_one_hot[i][1] = 1
    #     elif ground_truth[i] == 1:
    #         ground_truth_one_hot[i][2] = 1

    # shuffle the data
    # examples now holds array of all examples; ground_truth holds array of all labels
    # randomize the examples
    p = np.random.permutation(len(ground_truth_one_hot[:, 0]))
    random_examples = examples[p]
    #TODO: do non-linear feature transform here
    random_labels = ground_truth_one_hot[p]
    return random_examples, random_labels, movies

def build_model():
    #  model_type:  BNN for Bayesian network, FC for fully-connected/dense/linear model
    if args.model_type == 'BNN':
        model = bnn.BNN(25, 1, lr=0.001)
    elif args.model_type == 'FC':
        model = simple_model.FC(25, 1)
    return model

def compute_error(model, data, labels):
    error = 0.0
    for i in range(len(labels)):
        inputs = data[i, :]
        true_out = labels[i, :]
        inputs = inputs.reshape((1, len(inputs)))
        inputs = Variable(torch.from_numpy(inputs), volatile=True).float()
        output = model(inputs)
        error += ((true_out - output[0].data.numpy())**2.).sum() / len(true_out)
    return error / len(labels)

def compute_accuracy(model, data, labels):
    correct = 0.0
    for i in range(len(labels)):
        inputs = data[i, :]
        true_out = labels[i, :]
        inputs = inputs.reshape((1, len(inputs)))
        inputs = Variable(torch.from_numpy(inputs), volatile=True).float()
        output = model(inputs)
        if true_out == int(np.round(output[0].data.numpy())):
            correct += 1.
        # error += ((true_out - output[0].data.numpy())**2.).sum() / len(true_out)
    return correct / len(labels)

def train(model, data, labels):
    num_batches = int(np.floor(len(labels) / args.batch_size))

    e = 0
    while e < args.epochs:
        b = 0
        while b < num_batches:
            # if b % 100 == 0:
            # print ("Batch: ", b, " of ", num_batches)

            ind_start = b*args.batch_size
            ind_end = (b+1)*args.batch_size

            inputs = data[ind_start:ind_end,:]
            targets = labels[ind_start:ind_end,:]
            loss = model.train(inputs, targets)
            b += 1
        if e % 10 == 0:
            print ("Epoch: ", e, compute_error(model, data, labels), compute_accuracy(model, data, labels))
            # save the Model
            torch.save(model, 'models/model_' + args.model_type + '_epoch_'+str(e)+'.pth')

        e += 1



def compute_vpi(model, user_tag, movies):
    """
    Compute value of perfect information for each movie...
    """
    max_kl = -np.inf
    max_kl_movie = None
    max_kl_target = None
    for i in range(len(movies)):
        if i % 100 == 0:
            print ("Checking KL for movie ", i)
        m = movies[i]
        # print (m.shape)
        # input("")
        sample = np.concatenate((user_tag, m))[np.newaxis,:]
        for j in [0, 1]:
            target = np.array([[j]])
            # print (sample.shape, target.shape)
            kldiv = model.fast_kl_div(sample, target)

            # model.save_old_params()
            # model.train(sample, target)
            # kldiv = model.info_gain().data[0]
            # model.reset_to_old_params()

            # print ("KL: ", kldiv)
            # input("")

            if kldiv > max_kl:
                max_kl = kldiv
                max_kl_movie = m
                max_kl_target = target
                print ("Max KL: ", max_kl, list(max_kl_movie), float(max_kl_target))

    print ("Max KL: ", max_kl, list(max_kl_movie), float(max_kl_target))

if __name__ == '__main__':
    # pretrain model
    data, labels, movies = load_data()
    if args.load_model != '':
        model = torch.load(args.load_model)
    else:
        # create model given type
        model = build_model()
        # train
        train(model, data, labels)

    compute_vpi(model, data[0][0:5], movies)
