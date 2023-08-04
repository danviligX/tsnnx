import pickle
with open('data/raw/01.pickle','rb') as path:
    data = pickle.load(path)
    path.close()