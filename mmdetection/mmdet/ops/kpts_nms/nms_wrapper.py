import numpy as np
import torch

# from . import kpts_nms_cuda
from . import kpts_nms_cpu


def kpts_nms(kp_predictions, dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.
    """
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
        kp_predictions_th = kp_predictions
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_th = torch.from_numpy(dets).to(device)
        kp_predictions_th = torch.from_numpy(kp_predictions).to(device)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    # execute cpu or cuda nms
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        """
        if dets_th.is_cuda:
            inds = kpts_nms_cuda.nms(dets_th, iou_thr)
        else:
            inds = kpts_nms_cpu.nms(dets_th, iou_thr)
        """
        # Current only support cpu version
        inds = kpts_nms_cpu.kpts_nms(kp_predictions_th, dets_th, iou_thr)
    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds