import csv
import argparse
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='datafile_ml100k.npz', help='data file')
parser.add_argument('--recs-path', type=str, default='', help='recs npz file')
parser.add_argument('--model-type', type=str, default='BNN', help='model type, FC or BNN')
parser.add_argument('--batch-size', type=int, default=128, help='training batch size')
parser.add_argument('--epochs', type=int, default=10000, help='training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--eta', type=float, default=0.1, help='expl param')
parser.add_argument('--load-model', type=str, default='', help='which model to load')
parser.add_argument('--user', type=int, default=1, help='Which user to train on')
parser.add_argument('--nusers', type=int, default=5, help='num users to use')
parser.add_argument('--use-kernel', action='store_true')
parser.add_argument('--use-non-lin', action='store_true')
parser.add_argument('--use-fake-user', action='store_true')
parser.add_argument('--use-default-recs', action='store_true')
parser.add_argument('--use-user-tag', action='store_true', help='Note: not compatible with fake user')
parser.add_argument('--vpi', action='store_true')
parser.add_argument('--tag', type=str, help='exp tag')
args = parser.parse_args()


# script: python plotting/plot_learning.py --data-path models/bnn_lin_fake_user_learn_curve/train.csv,models/bnn_lin_kernel_fake_user_learn_curve/train.csv,models/bnn_nonlin_fake_user_learn_curve/train.csv,models/bnn_nonlin_kernel_fake_user_learn_curve/train.csv

# multi user: python plotting/plot_learning.py --data-path models/bnn_lin_multi_2user_learn_curve_sd01/train.csv,models/bnn_lin_kernel_multi_2user_learn_curve_sd01/train.csv,models/bnn_nonlin_multi_2user_learn_curve_sd01/train.csv,models/bnn_nonlin_kernel_multi_2user_learn_curve_sd01/train.csv

# get the data from the datafile_ml100k
datafile = args.data_path

N_USERS = 943
N_MOVIES = 1682

if args.use_fake_user:
    args.user = N_USERS
    N_USERS += 1


# load user data
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
    fake_user_did_like = [49, 88, 95, 120, 134, 171, 175, 180, 182, 221, \
        1, 28, 78, 116, 117, 143, 160, 173, 209, 225]
    fake_user_did_not_like = [370, 377, 388, 401, 415, 420, 426, 457, 482, 483, \
        40, 66, 71, 89, 138, 150, 152, 157, 162, 167]

    if args.use_user_tag:
        dim = 25
    else:
        dim = 20
    # add one for fake user
    example_array = np.zeros([(N_USERS) * N_MOVIES, dim])
    labels = np.zeros([(N_USERS) * N_MOVIES, 1])
    index = 0
    for i in range(N_USERS):
        for j in range(N_MOVIES):
            # first 1682 rows of batch correspond to first user;
            #  second 1682 rows of batch correspond to second user, etc.
            # example_array[index, 0:5] = Users[i, :]
            # example_array[index, 5:25] = Items[j, :]

            if args.use_user_tag:
                example_array[index,0:20] = Items[j, :]
                if i == N_USERS-1: # make fake user
                    if j in fake_user_did_like:
                        labels[index, 0] = 1.0
                    elif j in fake_user_did_not_like:
                        labels[index, 0] = 0.01
                else:
                    example_array[index,20:25] = Users[i, :]
                    labels[index, 0] = Ratings[i, j]
            else:
                example_array[index,:] = Items[j, :]
                if i == N_USERS-1: # make fake user
                    if j in fake_user_did_like:
                        labels[index, 0] = 1.0
                    elif j in fake_user_did_not_like:
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
    movies = example_array[0:N_MOVIES,0:20] # 5:25]

    # debug
    print("first row of labels: ", labels[0:N_MOVIES, 0])
    # examples & targets
    examples = example_array[args.user*N_MOVIES:(args.user+args.nusers)*N_MOVIES,:]
    ground_truth = labels[args.user*N_MOVIES:(args.user+args.nusers)*N_MOVIES]
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
    return random_examples, random_labels, movies, MovieTitles

def load_recs():
    recs = np.load(args.recs_path)
    for var in recs:
        return recs[var]

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_tsne(data, labels, recs, movies):
    # add recs to data
    data = np.concatenate((data, recs), axis=0)
    colors = ['blue' if l == 1 else 'red' for l in np.nditer(labels)]
    for i in range(len(recs)):
        colors.append('black')
    X_embedded = TSNE(perplexity=30, n_components=2).fit_transform(data)

    fig = plt.figure(figsize=(4, 3))
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c=colors)
    plt.tight_layout()

    blue_patch = mpatches.Patch(color='blue', label='Like')
    red_patch = mpatches.Patch(color='red', label='Dislike')
    black_patch = mpatches.Patch(color='black', label='Recs')
    legend = plt.legend(bbox_to_anchor=(0.15, 0.15), handles=[blue_patch, red_patch, black_patch])
    legend.get_frame().set_facecolor('none')
    plt.axis('off')
    plt.savefig('plot.png', format='png')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    data, labels, movies, titles = load_data()
    recs = load_recs()
    plot_tsne(data, labels, recs, movies)
