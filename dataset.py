#
# dataset.py
#  Python script to read in MovieLens dataset
#
import pandas
from scipy.sparse import csr_matrix
import numpy as np
from datetime import datetime
import time
#
# MovieLens dataset 100k
#   Contain ratings from 943 users (one row per user) and 1682 movies (one col per movie)
#   Total number of ratings:  100k, each rating from 1-5
#   Each user has rated at least 20 movies
#   ua.base file in /ml-100k/ dir is
# Ratings data
data_shape = (943, 1682)
data_dir = '../data/ml-100k/'
#  turn off reading all ratings with switch
ratings_switch = True
if ratings_switch:
    # get data from datafile
    df = pandas.read_csv(data_dir + "ua.base", sep="\t", header=-1)
    values = df.values
    # adjust indices to translate from 'index-starting-at-one' to 'index-starting-at-zero'
    values[:, 0:2] -= 1
    X_train = csr_matrix((values[:, 2], (values[:, 0], values[:, 1])), dtype=np.float, shape=data_shape)
    print "size of X_train: ", np.shape(X_train)
    #
    # debug
    print "first row of X_train: ", X_train[0, :]
    """
    print "X_train[0, 0]: ", X_train[0, 0]
    print "X_train[0, 26]: ", X_train[0, 26]
    print "X_train[0, 27]: ", X_train[0, 27]
    print "X_train[0, 280]: ", X_train[0, 280]
    print "X_train[0, 1000]: ", X_train[0, 1000]
    print "X_train[0, 2000]: ", X_train[0, 2000]
    """
    #  run through data and adjust ratings:
    #  Rating = 1 or 2:     change to -1
    #  Rating = 3:          change to 0
    #  Rating = 4 or 5:     change to 1
    Ratings = np.zeros([943, 1682])
    for i in range(943):
        for j in range(1682):
            example = X_train[i, j];
            if example == 1.0 or example == 2.0:
                Ratings[i, j] = -1
            elif example == 3.0:
                Ratings[i, j] = 0.01    # 0.01 signifies rated, but neutral; 0 signifies no rating
            elif example == 4.0 or example == 5.0:
                Ratings[i, j] = 1
    #
    # debug
    print "size of Ratings: ", np.shape(Ratings)
    print "first row of Ratings: ", Ratings[:, 0]
    rating_file = 'ratings_file.npz'
    np.savez(rating_file, Ratings=Ratings)
    print "Ratings file written"
#
#
# User information
#
#  turn off reading all users with switch
users_switch = True
if users_switch:
    # get data from datafile
    data_shape = (943, 1)
    df = pandas.read_csv(data_dir + "u.user", sep="\t", header=-1)
    values = df.values
    # debug
    print "shape of user values: ", np.shape(values)
    print "first row of values: ", values[0, :]
    AllUserInfo = []
    occ_list = []
    for i in range(943):
        print "i = ", i
        row = values[i, 0]
        print "row = ", row
        user_data = row.split('|')
        userid = int(user_data[0])   # userid: integer from 1 to 943
        userage = int(user_data[1])/99.0     # userage: float from 0 to 1
        usergender = 1
        if user_data[2] == 'M':
            usergender = 0      # usergender:  1=female, 0=male
        if user_data[3] not in occ_list:
            occ_list.append(user_data[3])   # string indicating occupation
        occ_index = 0
        for test in occ_list:
            if user_data[3] == test:
                userocc = occ_index
            occ_index = occ_index + 1       # userocc:  integer index of occupation in occ_list
        try:
            userzip = int(user_data[4])
        except:
            userzip = 0
            print "translated " + user_data[4] + " to 0"
        userzipcode = int(userzip)/99999.0      # userzip:  float from 0 to 1
        User_row = [userid, userage, usergender, userocc, userzipcode]
        AllUserInfo.append(User_row)

    UserInfo = np.array(AllUserInfo)
    # debug
    print "shape of UserInfo: ", np.shape(UserInfo)
    print "first row of UserInfo: ", UserInfo[0, :]
    # save the User data in array
    users_file = 'users_file.npz'
    np.savez(users_file, UserInfo=UserInfo)
#
#
#
# Item/Movie information
#
#  turn off reading all items with switch
items_switch = True
if items_switch:
    # get data from datafile
    data_shape = (1682, 1)
    df = pandas.read_csv(data_dir + "u.item", sep="\t", header=-1)
    values = df.values
    # debug
    print "shape of item values: ", np.shape(values)
    print "first row of items: ", values[0, :]
    AllItemInfo = []
    title_list = []
    #
    for i in range(1682):
        print "i = ", i
        row = values[i, 0]
        print "row = ", row
        item_data = row.split('|')
        itemid = int(item_data[0])   # itemid: integer from 1 to 1682
        # title_list.append(user_data[1])   # title:  ignore - use itemid
        itemdate = item_data[2]
        # for some movies, no itemdate is provided, so make sure it is there before converting
        if itemdate:
            d = datetime.strptime(itemdate, "%d-%b-%Y")
            itemdate = time.mktime(d.timetuple())   # itemdate: float num seconds after 1/1/70
        else:
            itemdate = 0.0
        # itemvrd =         # video release date:  ignore - use release date
        # URL               # ignore URL
        # unknown           # ignore unknown
        print "itemdate is: ", itemdate
        Item_row = [int(itemid), int(itemdate)]
        for j in range(6, 24):
            Item_row.append(int(item_data[j]))
        #
        print "shape of Item_row is: ", np.shape(Item_row)
        AllItemInfo.append(Item_row)

    ItemInfo = np.array(AllItemInfo)
    # debug
    print "shape of ItemInfo: ", np.shape(ItemInfo)
    print "first row of ItemInfo: ", ItemInfo[0, :]
    # save the Item data in array
    items_file = 'items_file.npz'
    np.savez(items_file, ItemInfo=ItemInfo)
#
