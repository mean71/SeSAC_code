import os
import pickle 

def create_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def save(obj, path):
    pickle.dump(obj, open(path, 'wb+'))

def load(obj, path):
    return pickle.load(open(path, 'rb'))