import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm, trange


test_seqs = [
    'downtown_arguing_00',
    'downtown_bar_00',
    'downtown_bus_00',
    'downtown_cafe_00',
    'downtown_car_00',
    'downtown_crossStreets_00',
    'downtown_downstairs_00',
    'downtown_enterShop_00',
    'downtown_rampAndStairs_00',
    'downtown_runForBus_00',
    'downtown_runForBus_01',
    'downtown_sitOnStairs_00',
    'downtown_stairs_00',
    'downtown_upstairs_00',
    'downtown_walkBridge_01',
    'downtown_walking_00',
    'downtown_walkUphill_00',
    'downtown_warmWelcome_00',
    'downtown_weeklyMarket_00',
    'downtown_windowShopping_00',
    'flat_guitar_01',
    'flat_packBags_00',
    'office_phoneCall_00',
    'outdoors_fencing_01',
]


def pw3d_extract(dataset_path='./', out_path='./'):

    # scale factor
    scaleFactor = 1

    data_list = []

    # get a list of .pkl files in the directory
    files = [os.path.join(dataset_path, 'sequenceFiles', seq+'.pkl') for seq in test_seqs]
    # go through all the .pkl files
    for filename in tqdm(files):
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            smpl_pose = data['poses']
            smpl_betas = data['betas']
            poses2d = data['poses2d']
            global_poses = data['cam_poses']
            genders_frame = data['genders']
            valids_frame = np.array(data['campose_valid']).astype(np.bool)
            num_people = len(smpl_pose)
            num_frames = len(smpl_pose[0])
            seq_name = str(data['sequence'])
            img_names = np.array(['imageFiles/' + seq_name + '/image_%s.jpg' % str(i).zfill(5) for i in range(num_frames)])
            
            for i in range(len(img_names)):
                imgnames_, parts_, valids_, bboxes_ = [], [], [], []
                poses_, shapes_, genders_ = [], [], []
                for j in range(num_people):
                    valid = valids_frame[j][i]
                    if not valid:
                        continue
                    betas = np.tile(smpl_betas[j][:10].reshape(1,-1), (num_frames, 1))
                    beta = betas[i]
                    imgname = img_names[i]
                    gender = genders_frame[j]
                    part = np.array(poses2d)[j][i].T
                    valid_part = part[part[:,2]>0,:]
                    if np.sum(valid_part) == 0:
                        bbox = np.array([0,0,0,0])
                    else:
                        bbox = np.array([min(valid_part[:,0]), min(valid_part[:,1]),
                                         max(valid_part[:,0]), max(valid_part[:,1])])
                    extrinsics = global_poses[i][:3,:3]
                    pose = np.array(smpl_pose)[j][i]
                    pose[:3] = cv2.Rodrigues(np.dot(extrinsics, cv2.Rodrigues(pose[:3])[0]))[0].T[0]
                    imgnames_.append(imgname)
                    valids_.append(valid)
                    parts_.append(part)
                    bboxes_.append(bbox)
                    poses_.append(pose)
                    shapes_.append(beta)
                    genders_.append(gender)

                if len(imgnames_) == 0:
                    continue
                parts = np.array(parts_)
                valids = np.array(valids_)
                bboxes = np.array(bboxes_)
                poses = np.array(poses_)
                shapes = np.array(shapes_)
                genders = np.array(genders_)
                data_dict = {'filename': imgnames_[0], 'bboxes': bboxes, 'poses': poses, 'parts': parts,
                             'shapes': shapes, 'genders': genders, 'valids': valids}
                data_list.append(data_dict)

    with open('./rcnn/test.pkl', 'wb') as f:
        pickle.dump(data_list, f)


if __name__ == "__main__":
    # pw3d_extract()
    data = pickle.load(open('./rcnn/test.pkl', 'rb'))
    for i in trange(len(data)):
        data[i]['shape'] = data[i].pop('shapes').astype(np.float32).reshape(-1, 10)
        data[i]['pose'] = data[i].pop('poses').astype(np.float32).reshape(-1, 72)
        data[i]['bboxes'] = data[i].pop('bboxes').astype(np.float32).reshape(-1, 4)
        data[i].pop('parts')
        valids = data[i].pop('valids')
        assert valids.min() == True
    with open('./rcnn/test.pkl', 'wb') as f:
        pickle.dump(data, f)