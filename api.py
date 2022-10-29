import os
from .recDataLoader import dataLoader
from .sim_matrix import similarity_matrix
from .recom import itemCF
import pickle
root_dir = os.path.dirname(os.path.realpath(__file__))
print (root_dir)
test_input = {
    'table':{
        'user1':{
                'wallhaven-o3w1ep.png': 6,
                'wallhaven-q2v2vl.jpg': 3,
                'wallhaven-k76586.jpg': 1,
                'wallhaven-0pr3lm.jpg': 6,
                'wallhaven-0jk6wp.jpg': 3,
                'wallhaven-p8j79j.jpg': 1,
            },
        'user2':{
            
        }
    }
}

def getUserRating(user, table):
    rating_dict = {}
    table = table[user]
    for item in table:
        if item in table.keys():
            rating_dict[item] = int(table[item])
        else:
            rating_dict[item] = 0
    return rating_dict

def rec_init(objects_path='./objects', filelist_path='./filelist/filelist.csv', img_dir='./imgs', featureMap_path='./features/features.json'):
    dl, sm, cf = None, None, None
    try:
        with open(os.path.join(root_dir,objects_path, 'dl.pkl'), 'rb') as f:
            print('loading dl')
            dl = pickle.load(f)
    except:
            print('dumping dl')
            dl = dataLoader(filelist_path, img_dir, featureMap_path)
            with open(os.path.join(objects_path, 'dl.pkl'), 'wb') as f:
                pickle.dump(dl, f)
            
    try:
        with open(os.path.join(root_dir,objects_path, 'sm.pkl'), 'rb') as f:
            print('loading sm')
            sm = pickle.load(f)
    except:
        print('dumping sm')
        sm = similarity_matrix(dl)
        with open(os.path.join(objects_path, 'sm.pkl'), 'wb') as f:
            pickle.dump(sm, f)
            
    try:
        with open(os.path.join(root_dir,objects_path, 'cf.pkl'), 'rb') as f:
            print('loading cf')
            cf = pickle.load(f)
    except:
        print('dumping cf')
        cf = itemCF(dl, sm)
        with open(os.path.join(root_dir,objects_path, 'cf.pkl'), 'wb') as f:
            pickle.dump(cf, f)
    return cf