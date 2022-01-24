// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/extension.h>

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")

at::Tensor kpts_nms_cuda(const at::Tensor kp_predictions, const at::Tensor boxes, float kpts_nms_overlap_thresh);

at::Tensor kpts_nms(const at::Tensor& kp_predictions, const at::Tensor& dets, const float threshold) {
  CHECK_CUDA(dets);
  CHECK_CUDA(kp_predictions);
  if (dets.numel() == 0)
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  return kpts_nms_cuda(kp_predictions, dets, threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kpts_nms", &nms, "non-maximum suppression");
}