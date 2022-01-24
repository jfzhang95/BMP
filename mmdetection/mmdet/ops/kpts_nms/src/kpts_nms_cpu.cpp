// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/extension.h>
#include <iostream>
#include <typeinfo>
float compute_oks(const at::Tensor& src_keypoints_2d, const at::Tensor& dst_keypoints_2d, const float area){
  float sigmas[24] = {.89, .87, 1.07, 1.07, .87, .89, .62, .72, .79, .79, .72, .62, .26, .26,
        1.07, .79, .79, .26, .26, .26, .25, .25, .25, .25};

  auto x1_t = src_keypoints_2d.select(1, 0).contiguous();
  auto y1_t = src_keypoints_2d.select(1, 1).contiguous();
  auto x2_t = dst_keypoints_2d.select(1, 0).contiguous();
  auto y2_t = dst_keypoints_2d.select(1, 1).contiguous();

  auto x1 = x1_t.data<float>();
  auto y1 = y1_t.data<float>();
  auto x2 = x2_t.data<float>();
  auto y2 = y2_t.data<float>();

  float e = 0.0f;
  auto n = dst_keypoints_2d.size(0);
  for(int64_t i = 0;i < n;i++){
    auto dx = x1[i] - x2[i];
    auto dy = y1[i] - y2[i];
    auto tmp = std::exp((-(dx * dx + dy * dy) / (sigmas[i] * sigmas[i] * 4 / 100) / (area + 1e-8) / 2));
    e += tmp;
  }
  return e / n;
}

template <typename scalar_t>
at::Tensor kpts_nms_cpu_kernel(const at::Tensor& kp_predictions, const at::Tensor& dets, const float threshold) {
  AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!kp_predictions.type().is_cuda(), "kp_predictions must be a CPU tensor");

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  }

  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();
  auto scores = dets.select(1, 4).contiguous();

  at::Tensor areas_t = (x2_t - x1_t + 1) * (y2_t - y1_t + 1);

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t =
      at::zeros({ndets}, dets.options().dtype(at::kByte).device(at::kCPU));

  auto suppressed = suppressed_t.data<uint8_t>();
  auto order = order_t.data<int64_t>();

  auto areas = areas_t.data<scalar_t>();
  // auto kp_predictions = _kp_predictions.data<scalar_t>();

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];

    if (suppressed[i] == 1) continue;
    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1) continue;
      auto ovr = compute_oks(kp_predictions[i], kp_predictions[j], iarea);
      if (ovr >= threshold) suppressed[j] = 1;
    }
  }
  return at::nonzero(suppressed_t == 0).squeeze(1);
}

at::Tensor kpts_nms(const at::Tensor& kp_predictions, const at::Tensor& dets, const float threshold) {
  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(dets.type(), "nms", [&] {
    result = kpts_nms_cpu_kernel<scalar_t>(kp_predictions, dets, threshold);
  });
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kpts_nms", &kpts_nms, "non-maximum suppression");
}