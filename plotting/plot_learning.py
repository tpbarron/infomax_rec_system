import csv
import argparse
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
import numpy as np

# script: python plotting/plot_learning.py --data-path models/bnn_lin_fake_user_learn_curve/train.csv,models/bnn_lin_kernel_fake_user_learn_curve/train.csv,models/bnn_nonlin_fake_user_learn_curve/train.csv,models/bnn_nonlin_kernel_fake_user_learn_curve/train.csv

# multi user: python plotting/plot_learning.py --data-path models/bnn_lin_multi_2user_learn_curve_sd01/train.csv,models/bnn_lin_kernel_multi_2user_learn_curve_sd01/train.csv,models/bnn_nonlin_multi_2user_learn_curve_sd01/train.csv,models/bnn_nonlin_kernel_multi_2user_learn_curve_sd01/train.csv

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='', help='data file')
args = parser.parse_args()

labels = ['Linear', 'Lin Kernel', 'Non-Lin', 'Non-Lin Kernel']
ylabels = ['MSE', 'Accuracy']

def smooth(x, window_len=25, window='hanning'):
    """
    NOTE: FROM SCIPY TUTORIAL

    smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError ("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

# epoch, batch ,error, acc
def read_data(path):
    errors, acc = [], []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            errors.append(float(row[2]))
            acc.append(float(row[3]))
    return errors, acc

def plot_data(data):
    fig = plt.figure(figsize=(4, 3))
    ax = plt.gca()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    for i in range(len(data)):
        d = data[i]
        d = np.array(d)
        d = smooth(d)[:5000] #len(d)]
        plt.plot(d, label=labels[i], alpha=0.75)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot.png', format='png')
    # plt.show()

if __name__ == '__main__':
    paths = (args.data_path).split(',')
    data = []
    for path in paths:
        errors, acc = read_data(path)
        data.append((errors, acc))

    errors, accs = zip(*data)
    plot_data(errors)
    input("next?")
    plot_data(accs)
