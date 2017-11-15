import csv
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='', help='data file')
args = parser.parse_args()


def read_data():
    kls = []
    with open(args.data_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # print (row)
            # input("")
            kls.append(float(row[1]))
    return kls

if __name__ == '__main__':
    kls = read_data()
    plt.plot(kls)
    plt.show()
