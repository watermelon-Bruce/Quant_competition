import pickle
def pickle_store(avilable,store_road):
    file = open(store_road, 'wb')
    pickle.dump(avilable, file)
    file.close()

def pickle_read(read_road):
    with open(read_road, 'rb') as file:
        avilable = pickle.load(file)
    return avilable