# create_datafile.py
import numpy as np
#
with np.load('ratings_file.npz') as a:
    Ratings = a['Ratings']
#
with np.load('users_file.npz') as b:
    Users = b['UserInfo']
#
with np.load('items_file.npz') as c:
    Items = c['ItemInfo']
#
# save into new file
np.savez('datafile_ml100k.npz', Ratings=Ratings, Users=Users, Items=Items)
#
