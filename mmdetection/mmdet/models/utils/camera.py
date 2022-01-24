import torch
import numpy as np


class PerspectiveCamera(object):
    def __init__():  
        pass

    def __call__(points, batch_size, rotation, 
                 translation, focal_length, camera_center):
        """
        This function computes the perspective projection of a set of points.
        Input:
            points (bs, N, 3): 3D points
            rotation (bs, 3, 3): Camera rotation
            translation (bs, 3): Camera translation
            focal_length (bs,) or scalar: Focal length
            camera_center (bs, 2): Camera center
        """                      
        K = torch.zeros([batch_size, 3, 3], device=points.device)
        K[:,0,0] = focal_length
        K[:,1,1] = focal_length
        K[:,2,2] = 1.
        K[:,:-1, -1] = camera_center

        # Transform points
        points = torch.einsum('bij,bkj->bki', rotation, points)
        points = points + translation.unsqueeze(1)

        # Apply perspective distortion
        projected_points = points / points[:,:,-1].unsqueeze(-1)

        # Apply camera intrinsics
        projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

        return projected_points[:, :, :-1]