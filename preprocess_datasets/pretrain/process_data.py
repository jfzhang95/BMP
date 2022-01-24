import pickle
import os.path as osp
import numpy as np
from tqdm import tqdm

pkl_file = 'mpi_inf_3dhp/train.pkl'
data = pickle.load(open(pkl_file, 'rb'))
for i, data_i in tqdm(enumerate(data)):
    filename = data_i['filename'].split('/')[-1]
    bboxes = np.array(data_i['bboxes'])
    kpts2d = np.array(data_i['kpts2d'])
    kpts3d = np.array(data_i['kpts3d'])
    assert bboxes.shape == (1, 4)
    assert kpts2d.shape == (1, 24, 3)
    assert kpts3d.shape == (1, 24, 4)
    data[i]['filename'] = filename
    data[i]['bboxes'] = bboxes
    data[i]['kpts2d'] = kpts2d
    data[i]['kpts3d'] = kpts3d

with open(pkl_file, 'wb') as f:
    pickle.dump(data, f)