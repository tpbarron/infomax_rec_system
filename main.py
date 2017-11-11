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

N_USERS = 943
N_MOVIES = 1682

def load_data():
    with np.load(datafile) as f:
        Ratings = f['Ratings']  # 943 users x 1682 movies
        Users = f['Users']      # 943 users x 5 fields
        Items = f['Items']      # 1682 movies x 20 fields
        MovieTitles = f['MovieTitles']
    # debug
    print("first row of Ratings: ", Ratings[0, :])
    print("first Movie Title (should be Toy Story): ", MovieTitles[0])

    #   set up the array of examples
    example_array = np.zeros([N_USERS * N_MOVIES, 25])
    labels = np.zeros([N_USERS * N_MOVIES, 1])
    index = 0
    for i in range(N_USERS):
        for j in range(N_MOVIES):
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

    movies = example_array[0:N_MOVIES, 5:25]

    # debug
    print("first row of labels: ", labels[0:N_MOVIES, 0])
    # examples & targets
    examples = example_array[0:N_MOVIES,:]
    ground_truth = labels[0:N_MOVIES]
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
    #TODO: do non-linear feature transform here?
    random_labels = ground_truth_one_hot[p]
    return random_examples, random_labels, movies, MovieTitles

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

def train(model, data, labels, epochs, retrain=0):
    num_batches = int(np.floor(len(labels) / args.batch_size))

    e = 0
    while e < epochs:
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
            print ("Epoch: ", e, ", retrain: ", retrain, compute_error(model, data, labels), compute_accuracy(model, data, labels))
            # save the Model
            torch.save(model, 'models/model_' + args.model_type + '_epoch_'+str(e)+'_retrain_'+str(retrain)+'.pth')

        e += 1



def compute_vpi(model, user_tag, movies):
    """
    Compute value of perfect information for each movie...
    """
    max_kl = -np.inf
    max_kl_movie = None
    max_kl_target = None
    for i in range(N_MOVIES):
        if i % 100 == 0:
            print ("Checking KL for movie ", i)
        m = movies[i]
        # print (m.shape)
        # input("")
        sample = np.concatenate((user_tag, m))[np.newaxis,:]
        for j in [0]: #, 1]:
            # Looking for max KL if user doesn't like.
            # This means that we think the user should like it a lot!
            target = np.array([[j]])
            kldiv = model.fast_kl_div(sample, target)

            # model.save_old_params()
            # model.train(sample, target)
            # kldiv = model.info_gain().data[0]
            # model.reset_to_old_params()

            # print ("KL: ", kldiv)
            # input("")

            if kldiv >= max_kl:
                max_kl = kldiv
                max_kl_movie = m
                max_kl_target = target
                print ("Max KL: ", max_kl, list(max_kl_movie), float(max_kl_target))

    print ("Max KL: ", max_kl, list(max_kl_movie), float(max_kl_target))
    return max_kl, max_kl_movie, max_kl_target

def get_movie_name(titles, id):
    return str(titles[id])

def print_user_prefs(data, labels, titles):
    """
    Given a subset of data for a specific user, print the movies the user likes and doesn't like"""
    for i in range(len(data)):
        if labels[i] == 0:
            print ("User does not like: ", get_movie_name(titles, int(data[i][5]*N_MOVIES)))
    for i in range(len(data)):
        if labels[i] == 1:
            print ("User likes: ", get_movie_name(titles, int(data[i][5]*N_MOVIES)))

if __name__ == '__main__':
    data, labels, movies, titles = load_data()
    print_user_prefs(data, labels, titles)
    # pretrain model
    if args.load_model != '':
        model = torch.load(args.load_model)
    else:
        # create model given type
        model = build_model()
        # train
        train(model, data, labels, args.epochs)

    itrs = 1
    while True:
        # alternate, recommendation, retraining
        kl, movie, target = compute_vpi(model, data[0][0:5], movies)
        resp = input("Do you like the movie " + get_movie_name(titles, int(movie[0]*N_MOVIES)) + "? Y/N. ")
        # concat new movie to dataset
        if resp == 'Y' or resp == 'y':
            new_label = np.array([[1.]])
        else:
            new_label = np.array([[0.]])
        # concat user portion with movie portion
        new_sample = np.concatenate((data[0][0:5], movie))[np.newaxis,:]
        # print (new_sample.shape)
        # print (new_label.shape)
        # print (data.shape)
        # print (labels.shape)
        data = np.concatenate((data, new_sample))
        labels = np.concatenate((labels, new_label))
        # print (data.shape)
        # print (labels.shape)
        # input("Next?")
        train(model, data, labels, epochs=100, retrain=itrs)
        itrs += 1
