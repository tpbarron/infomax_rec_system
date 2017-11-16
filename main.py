import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import polynomial_kernel
import pickle
import os

import torch
from torch.autograd import Variable

import bnn
import simple_model

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='datafile_ml100k.npz', help='data file')
parser.add_argument('--model-type', type=str, default='BNN', help='model type, FC or BNN')
parser.add_argument('--batch-size', type=int, default=128, help='training batch size')
parser.add_argument('--epochs', type=int, default=10000, help='training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--eta', type=float, default=0.1, help='expl param')
parser.add_argument('--load-model', type=str, default='', help='which model to load')
parser.add_argument('--user', type=int, default=1, help='Which user to train on')
parser.add_argument('--use-kernel', action='store_true')
parser.add_argument('--use-non-lin', action='store_true')
parser.add_argument('--use-fake-user', action='store_true')
parser.add_argument('--use-default-recs', action='store_true')
parser.add_argument('--vpi', action='store_true')
parser.add_argument('--tag', type=str, help='exp tag')
args = parser.parse_args()

log_dir = os.path.join('models/', args.tag)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
with open (os.path.join(log_dir, 'args_snapshot.pkl'), 'wb') as f:
    pickle.dump(args, f)

# get the data from the datafile_ml100k
datafile = args.data_path

N_USERS = 943
N_MOVIES = 1682

if args.use_fake_user:
    args.user = N_USERS

def kernel_transform(data):
    new_data = np.copy(data)

    # append squares
    data_sqrs = np.copy(data)
    data_sqrs = data_sqrs ** 2.
    new_data = np.concatenate((new_data, data_sqrs), axis=1)
    # print (new_data.shape)
    # pairs
    # 12, 13, ... 1n
    # num = sum(1 to n-1) = (n-1 * n-2) / 2 = 19*20 / 2 = 190
    for i in range(1, data.shape[1]-1):
        for j in range(i+1, data.shape[1]):
            # vec of col i * col j
            coli = data[:,i]
            colj = data[:,j]
            pair = coli * colj
            pair = pair.reshape(-1, 1)
            # print ("Pair: ", pair.shape)
            new_data = np.concatenate((new_data, pair), axis=1)

    # print ("new data: ", new_data.shape)
    return new_data

def load_data():
    with np.load(datafile) as f:
        Ratings = f['Ratings']  # 943 users x 1682 movies
        Users = f['Users']      # 943 users x 5 fields
        Items = f['Items']      # 1682 movies x 20 fields
        MovieTitles = f['MovieTitles']
    # debug
    print("first row of Ratings: ", Ratings[0, :])
    print("first Movie Title (should be Toy Story): ", MovieTitles[0])
    # debug info - get titles of all movies
    """
    for j in range(N_MOVIES):
        print("Index=%d Movie Title=%s" % (j, MovieTitles[j]))
    #
    exit()
    """
    #   set up the array of examples
    # example_array = np.zeros([N_USERS * N_MOVIES, 25])
    fake_user_did_like = [49, 88, 95, 120, 134, 171, 175, 180, 182, 221, \
        1, 28, 78, 116, 117, 143, 160, 173, 209, 225]
    fake_user_did_not_like = [370, 377, 388, 401, 415, 420, 426, 457, 482, 483, \
        40, 66, 71, 89, 138, 150, 152, 157, 162, 167]

    # add one for fake user
    example_array = np.zeros([(N_USERS+1) * N_MOVIES, 20])
    labels = np.zeros([(N_USERS+1) * N_MOVIES, 1])
    index = 0
    for i in range(N_USERS+1):
        for j in range(N_MOVIES):
            # first 1682 rows of batch correspond to first user;
            #  second 1682 rows of batch correspond to second user, etc.
            # example_array[index, 0:5] = Users[i, :]
            # example_array[index, 5:25] = Items[j, :]

            example_array[index,:] = Items[j, :]
            if i == N_USERS: # make fake user
                if j in fake_user_did_like:
                    # print ("TEST like")
                    labels[index, 0] = 1.0
                elif j in fake_user_did_not_like:
                    # print ("TEST not like")
                    labels[index, 0] = 0.01
            else:
                labels[index, 0] = Ratings[i, j]

            index += 1


    # normalize all data, doing this before selecting nonzero ensures consistent normalization
    scaler = MinMaxScaler()
    scaler.fit(example_array)
    example_array = scaler.transform(example_array)
    print (example_array.min(), example_array.max())

    # normalized movies
    movies = example_array[0:N_MOVIES,:]# 5:25]

    # debug
    print("first row of labels: ", labels[0:N_MOVIES, 0])
    # examples & targets
    examples = example_array[args.user*N_MOVIES:(args.user+1)*N_MOVIES,:]
    ground_truth = labels[args.user*N_MOVIES:(args.user+1)*N_MOVIES]
    nz = np.nonzero(ground_truth)[0]
    examples = examples[nz,:]
    ground_truth = ground_truth[nz]

    ground_truth_one_hot = np.zeros((len(ground_truth), 1))
    for i in range(len(ground_truth)):
        if ground_truth[i] == 0.01:
            print ("Negative")
            ground_truth_one_hot[i] = 0      # [0] means rating was 1-3
        elif ground_truth[i] == 1:
            print ("Positive")
            ground_truth_one_hot[i] = 1      # [1] means rating was 4-5

    # shuffle the data
    # examples now holds array of all examples; ground_truth holds array of all labels
    # randomize the examples
    p = np.random.permutation(len(ground_truth_one_hot[:, 0]))
    random_examples = examples[p]

    # do non-linear feature transform here?
    if args.use_kernel:
        random_examples = kernel_transform(random_examples)
    random_labels = ground_truth_one_hot[p]
    print ("Data / targets: ", random_examples.shape, random_labels.shape)
    # input("")
    return random_examples, random_labels, movies, MovieTitles

def build_model(input_dim=20, output_dim=1, n_batches=5):
    #  model_type:  BNN for Bayesian network, FC for fully-connected/dense/linear model
    if args.model_type == 'BNN':
        # 0.005
        model = bnn.BNN(input_dim, output_dim, lr=args.lr, n_batches=n_batches, nonlin=args.use_non_lin)
    elif args.model_type == 'FC':
        model = simple_model.FC(input_dim, output_dim, lr=args.lr, nonlin=args.use_non_lin)
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
        if true_out == 1 and output[0].data.numpy() >= 0.5:
            correct += 1
        elif true_out == 0 and output[0].data.numpy() < 0.5:
            correct += 1.
    return correct / len(labels)

def train(model, data, labels, epochs, retrain=0):
    num_batches = int(np.ceil(len(labels) / args.batch_size))

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
        if e % 100 == 0:
            print ("Epoch: ", e, ", retrain: ", retrain, compute_error(model, data, labels), compute_accuracy(model, data, labels))
            # save the Model
            torch.save(model, log_dir+'/model_' + args.model_type + '_epoch_'+str(e)+'_retrain_'+str(retrain)+'.pth')

        e += 1

    print ("Epoch: ", e, ", retrain: ", retrain, compute_error(model, data, labels), compute_accuracy(model, data, labels))
    # save the Model
    torch.save(model, log_dir+'/model_' + args.model_type + '_epoch_'+str(e)+'_retrain_'+str(retrain)+'.pth')


def compute_vpi(model, user_tag, movies, rated_ids):
    """
    Compute value of perfect information for each movie...

    TODO: don't permit recommending movie that user already rated?
    """
    max_kl = -np.inf
    max_pred = -np.inf
    max_kl_movie = None
    max_kl_target = None
    for i in range(N_MOVIES):
        # if i % 100 == 0:
            # print ("Checking KL for movie ", i)
        m = movies[i]
        if int(m[0]*N_MOVIES) in rated_ids:
            continue
        # print (m.shape)
        # input("")
        sample = m[np.newaxis,:]
        if args.use_kernel:
            sample = kernel_transform(sample)

        if args.vpi:
            # sample = np.concatenate((user_tag, m))[np.newaxis,:]
            for j in [0]: #, 1]:
                # Looking for max KL if user doesn't like.
                # This means that we think the user should like it a lot!
                target = np.array([[j]])
                kldiv = model.fast_kl_div(sample, target)

                if kldiv >= max_kl:
                    max_kl = kldiv
                    max_kl_movie = m
                    max_kl_target = target
                    print ("Max KL: ", max_kl, list(max_kl_movie), float(max_kl_target))
        else:
            # try something diff, use explicit trade off
            prediction = model(Variable(torch.from_numpy(sample), volatile=True).float()).data[0,0]
            # target = np.array([[1]])
            expected_kl = prediction * model.fast_kl_div(sample, np.array([[1]])) + (1-prediction) * model.fast_kl_div(sample, np.array([[0]]))
            # expected_kl = model.fast_kl_div(sample, np.array([[1]])) + model.fast_kl_div(sample, np.array([[0]]))
            # kldiv = model.fast_kl_div(sample, target)
            # print ("pred: ", prediction, ", kldiv: ", kldiv)
            reward = prediction + args.eta * expected_kl

            if reward >= max_pred:
                max_pred = reward
                # max_kl = kldiv
                max_kl = expected_kl
                max_kl_movie = m
                max_kl_target = -1
                # print ("Max KL: ", max_kl, list(max_kl_movie))

    print ("Max KL: ", max_kl, list(max_kl_movie), float(max_kl_target))
    return max_pred, max_kl, max_kl_movie, max_kl_target

def compute_default(model, user_tag, movies, rated_ids):
    """
    TODO: don't permit recommending movie that user already rated?
    """
    max_like = -np.inf
    max_like_movie = None
    for i in range(N_MOVIES):
        if i % 100 == 0:
            print ("Checking movie ", i)
        m = movies[i]
        if int(m[0]*N_MOVIES) in rated_ids:
            continue
        sample = m[np.newaxis,:]
        if args.use_kernel:
            sample = kernel_transform(sample)

        prediction = model(Variable(torch.from_numpy(sample), volatile=True).float()).data[0,0]
        if prediction >= max_like:
            max_like = prediction
            max_like_movie = m
            print ("Max Like: ", max_like, list(max_like_movie))

    print ("Max Like: ", max_like, list(max_like_movie))
    return max_like, max_like_movie


def get_movie_name(titles, id):
    return str(titles[id])

def print_user_prefs(data, labels, titles):
    """
    Given a subset of data for a specific user, print the movies the user likes and doesn't like
    """
    for i in range(len(data)):
        if labels[i] == 0:
            # print ("User does not like: ", get_movie_name(titles, int(data[i][5]*N_MOVIES)))
            print ("User does not like: ", list(data[i]), get_movie_name(titles, int(data[i][0]*N_MOVIES)))
    for i in range(len(data)):
        if labels[i] == 1:
            # print ("User likes: ", get_movie_name(titles, int(data[i][5]*N_MOVIES)))
            print ("User likes: ", list(data[i]), get_movie_name(titles, int(data[i][0]*N_MOVIES)))

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne(data, labels, movies):
    for i in [30]:
        print ("perplexity: ", i)
        X_embedded = TSNE(perplexity=25, n_components=2).fit_transform(data)
        colors = ['blue' if l == 1 else 'red' for l in np.nditer(labels)]
        plt.scatter(X_embedded[:,0], X_embedded[:,1], c=colors)
        plt.show()
        # input("Done")
        plt.close()

import csv

if __name__ == '__main__':
    data, labels, movies, titles = load_data()
    rated_ids = []
    for i in range(len(data)):
        rated_ids.append(int(data[i,0]*N_MOVIES))

    # plot_tsne(data, labels, movies)
    print_user_prefs(data, labels, titles)
    # input("")
    # pretrain model
    if args.load_model != '':
        model = torch.load(args.load_model)
    else:
        # create model given type
        model = build_model(input_dim=data.shape[1], n_batches=int(np.floor(len(labels) / args.batch_size)))
        # train
        train(model, data, labels, args.epochs)

    kl_file = open(os.path.join(log_dir, 'kls.txt'), 'w')
    writer = csv.writer(kl_file)
    itrs = 1
    while True:
        # alternate, recommendation, retraining
        # kl, movie, target = compute_vpi(model, data[0][0:5], movies)
        if args.use_default_recs:
            pred, movie = compute_default(model, None, movies, rated_ids)
            kl = -1
            target = -1
        else:
            pred, kl, movie, target = compute_vpi(model, None, movies, rated_ids)

        movie_id = int(movie[0]*N_MOVIES)
        movie_name = get_movie_name(titles, movie_id)
        # resp = input("Do you like the movie " + get_movie_name(titles, movie_id) + "? Y/N. ")
        #
        #
        print("Do you like the movie " + get_movie_name(titles, movie_id) + "? Y/N. ")
        resp = np.random.choice(['Y', 'N'])
        print("Response is " + resp)
        # concat new movie to dataset
        # resp = 'y' #if np.random.random() < 0.5 else 'n'
        if resp == 'Y' or resp == 'y':
            new_label = np.array([[1.]])
        else:
            new_label = np.array([[0.]])

        writer.writerow([str(pred), str(kl), np.array_str(movie, max_line_width=1000000), str(movie_id), str(movie_name)])
        kl_file.flush()

        # TODO: check whether sample is in data first, change label if needed
        # concat user portion with movie portion
        # new_sample = np.concatenate((data[0][0:5], movie))[np.newaxis,:]
        rated_ids.append(int(movie[0]*N_MOVIES))
        new_sample = movie[np.newaxis,:]
        data_index = np.argwhere(data[:, 0] == movie[0])
        print ("Existing data_index:", data_index)
        if len(data_index) > 0:
            labels[data_index[0][0]] = new_label
        else:
            if args.use_kernel:
                new_sample = kernel_transform(new_sample)
            data = np.concatenate((data, new_sample))
            labels = np.concatenate((labels, new_label))

        train(model, data, labels, epochs=100, retrain=itrs)
        itrs += 1

        if itrs > 10:
            break

    writer.close()
    kl_file.close()
