import os
from recDataLoader import dataLoader
from similarity_matrix import similarity_matrix
from recom import itemCF
import pickle

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

def rec_init(objects_path, filelist_path, img_dir, featureMap_path):
    dl, sm, cf = None, None, None
    try:
        with open(os.path.join(objects_path, 'dl.pkl'), 'rb') as f:
            print('loading dl')
            dl = pickle.load(f)
    except:
            dl = dataLoader(filelist_path, img_dir, featureMap_path)
            with open(os.path.join(objects_path, 'dl.pkl'), 'wb') as f:
                print('dumping dl')
                pickle.dump(dl, f)
            
    try:
        with open(os.path.join(objects_path, 'sm.pkl'), 'rb') as f:
            print('loading sm')
            sm = pickle.load(f)
    except:
        sm = similarity_matrix(dl)
        with open(os.path.join(objects_path, 'sm.pkl'), 'wb') as f:
            print('dumping sm')
            pickle.dump(sm, f)
            
    try:
        with open(os.path.join(objects_path, 'cf.pkl'), 'rb') as f:
            print('loading cf')
            cf = pickle.load(f)
    except:
        cf = itemCF(dl, sm)
        with open(os.path.join(objects_path, 'cf.pkl'), 'wb') as f:
            print('dumping cf')
            pickle.dump(cf, f)
    return cf

rating_dict = getUserRating('user1', test_input['table'])
cf = rec_init('./objects', './filelist/filelist.csv', './imgs', './features/features.json')
output = cf.recommend(top=20, rating_dict=rating_dict, shuffle=True)
print(output)