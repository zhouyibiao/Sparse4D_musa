
#include <ATen/ATen.h>
#include <ATen/musa/MUSAContext.h>
#include <musa.h>
#include <musa_runtime.h>

#include <ATen/musa/Atomic.muh>

#include <iostream>
#include <stdlib.h>
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"
#include <torch_musa/csrc/core/MUSAGuard.h>
#include <torch_musa/csrc/core/MUSAStream.h>


__device__ float bilinear_sampling(
    const float *&bottom_data, const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr
) {
  const int h_low = floorf(h_im);
  const int w_low = floorf(w_im);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = h_im - h_low;
  const float lw = w_im - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  const int w_stride = num_embeds;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}


__device__ void bilinear_sampling_grad(
    const float *&bottom_data, const float &weight,
    const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr,
    const float &grad_output,
    float *&grad_mc_ms_feat, float *grad_sampling_location, float *grad_weights) {
  const int h_low = floorf(h_im);
  const int w_low = floorf(w_im);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = h_im - h_low;
  const float lw = w_im - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  const int w_stride = num_embeds;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const float top_grad_mc_ms_feat = grad_output * weight;
  float grad_h_weight = 0, grad_w_weight = 0;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_mc_ms_feat + ptr1, w1 * top_grad_mc_ms_feat);
  }
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_mc_ms_feat + ptr2, w2 * top_grad_mc_ms_feat);
  }
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_mc_ms_feat + ptr3, w3 * top_grad_mc_ms_feat);
  }
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_mc_ms_feat + ptr4, w4 * top_grad_mc_ms_feat);
  }

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  atomicAdd(grad_weights, grad_output * val);
  atomicAdd(grad_sampling_location, width * grad_w_weight * top_grad_mc_ms_feat);
  atomicAdd(grad_sampling_location + 1, height * grad_h_weight * top_grad_mc_ms_feat);
}


__device__ void bilinear_sampling_grad_v2(
    const float *&bottom_data, const float &weight,
    const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr,
    const float &grad_output,
    float *&grad_mc_ms_feat, float *grad_sampling_location, float *grad_weights) {
  const int h_low = floorf(h_im);
  const int w_low = floorf(w_im);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = h_im - h_low;
  const float lw = w_im - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  const int w_stride = num_embeds;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const float top_grad_mc_ms_feat = grad_output * weight;
  float grad_h_weight = 0, grad_w_weight = 0;

  const int ptr1 = (h_low >= 0 && w_low >= 0) ? (base_ptr + h_low_ptr_offset + w_low_ptr_offset) : -1;
  const int ptr2 = (h_low >= 0 && w_high < width) ? (base_ptr + h_low_ptr_offset + w_high_ptr_offset) : -1;
  const int ptr3 = (h_high < height && w_low >= 0) ? (base_ptr + h_high_ptr_offset + w_low_ptr_offset) : -1;
  const int ptr4 = (h_high < height && w_high < width) ? (base_ptr + h_high_ptr_offset + w_high_ptr_offset) : -1;


  const float v1 = (ptr1 != -1) ? bottom_data[ptr1] : 0.0f;
  const float v2 = (ptr2 != -1) ? bottom_data[ptr2] : 0.0f;
  const float v3 = (ptr3 != -1) ? bottom_data[ptr3] : 0.0f;
  const float v4 = (ptr4 != -1) ? bottom_data[ptr4] : 0.0f;

  grad_h_weight -= (ptr1 != -1) ? hw * v1 : 0.0f;
  grad_h_weight -= (ptr2 != -1) ? lw * v2 : 0.0f;
  grad_h_weight += (ptr3 != -1) ? hw * v3 : 0.0f;
  grad_h_weight += (ptr4 != -1) ? lw * v4 : 0.0f;

  grad_w_weight -= (ptr1 != -1) ? hh * v1 : 0.0f;
  grad_w_weight += (ptr2 != -1) ? hh * v2 : 0.0f;
  grad_w_weight -= (ptr3 != -1) ? lh * v3 : 0.0f;
  grad_w_weight += (ptr4 != -1) ? lh * v4 : 0.0f;


  if (ptr1 != -1) {
    // grad_h_weight -= hw * v1;
    // grad_w_weight -= hh * v1;
    atomicAdd(grad_mc_ms_feat + ptr1, w1 * top_grad_mc_ms_feat);
  }

  if (ptr2 != -1) {
    // grad_h_weight -= lw * v2;
    // grad_w_weight += hh * v2;    
    atomicAdd(grad_mc_ms_feat + ptr2, w2 * top_grad_mc_ms_feat);
  }

  if (ptr3 != -1) {
    // grad_h_weight += hw * v3;
    // grad_w_weight -= lh * v3;
    atomicAdd(grad_mc_ms_feat + ptr3, w3 * top_grad_mc_ms_feat);
  }

  if (ptr4 != -1) {
    // grad_h_weight += lw * v4;
    // grad_w_weight += lh * v4;    
    atomicAdd(grad_mc_ms_feat + ptr4, w4 * top_grad_mc_ms_feat);
  }

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  atomicAdd(grad_weights, grad_output * val);
  atomicAdd(grad_sampling_location, width * grad_w_weight * top_grad_mc_ms_feat);
  atomicAdd(grad_sampling_location + 1, height * grad_h_weight * top_grad_mc_ms_feat);
}


__global__ void deformable_aggregation_kernel(
    const int64_t num_kernels,
    float* output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    const float weight = *(weights + idx / (num_embeds / num_groups));
    const int channel_index = idx % num_embeds;
    idx /= num_embeds;
    const int scale_index = idx % num_scale;
    idx /= num_scale;

    const int cam_index = idx % num_cams;
    idx /= num_cams;
    const int pts_index = idx % num_pts;
    idx /= num_pts;

    int anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int batch_index = idx % batch_size;
    idx /= batch_size;

    anchor_index = batch_index * num_anchors + anchor_index;
    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1) return;
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1) return;
    
    int cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    atomicAdd(
        output + anchor_index * num_embeds + channel_index,
        bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight
    );
}


__global__ void deformable_aggregation_kernel_v1(
    const int64_t num_kernels,
    float* output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups,
    at::musa::FastDivmod fastdiv1,
    at::musa::FastDivmod fastdiv2,
    at::musa::FastDivmod fastdiv3,
    at::musa::FastDivmod fastdiv4,
    at::musa::FastDivmod fastdiv5,
    at::musa::FastDivmod fastdiv6
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    uint32_t idx1, idx2, idx3, idx4, idx5, idx6;

    const float weight = *(weights + idx / (num_embeds / num_groups));
    uint32_t channel_index; // uint32_t
    fastdiv1(idx1, channel_index, idx);

    uint32_t scale_index; // uint32_t
    fastdiv2(idx2, scale_index, idx1);

    uint32_t cam_index; // uint32_t
    fastdiv3(idx3, cam_index, idx2);

    uint32_t pts_index; // uint32_t
    fastdiv4(idx4, pts_index, idx3);

    uint32_t anchor_index; // uint32_t
    fastdiv5(idx5, anchor_index, idx4);

    uint32_t batch_index; // uint32_t
    fastdiv6(idx6, batch_index, idx5);

    // const int channel_index = idx % num_embeds;
    // idx /= num_embeds;
    // const int scale_index = idx % num_scale;
    // idx /= num_scale;

    // const int cam_index = idx % num_cams;
    // idx /= num_cams;
    // const int pts_index = idx % num_pts;
    // idx /= num_pts;

    // int anchor_index = idx % num_anchors;
    // idx /= num_anchors;
    // const int batch_index = idx % batch_size;
    // idx /= batch_size;

    anchor_index = batch_index * num_anchors + anchor_index;
    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1) return;
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1) return;
    
    int cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    float new_data = bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight;
    atomicAdd(
        output + anchor_index * num_embeds + channel_index,
        new_data
    );
}


__global__ void deformable_aggregation_grad_kernel(
    const int64_t num_kernels,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    const float* grad_output,
    float* grad_mc_ms_feat,
    float* grad_sampling_location,
    float* grad_weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    const int weights_ptr = idx / (num_embeds / num_groups);
    const int channel_index = idx % num_embeds;
    idx /= num_embeds;
    const int scale_index = idx % num_scale;
    idx /= num_scale;

    const int cam_index = idx % num_cams;
    idx /= num_cams;
    const int pts_index = idx % num_pts;
    idx /= num_pts;

    int anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int batch_index = idx % batch_size;
    idx /= batch_size;

    anchor_index = batch_index * num_anchors + anchor_index;
    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1) return;
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1) return;
    
    const float grad = grad_output[anchor_index*num_embeds + channel_index];

    int cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    /* atomicAdd( */
    /*     output + anchor_index * num_embeds + channel_index, */
    /*     bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight */
    /* ); */
    const float weight = weights[weights_ptr];
    float *grad_weights_ptr = grad_weights + weights_ptr;
    float *grad_location_ptr = grad_sampling_location + loc_offset;
    bilinear_sampling_grad(
        mc_ms_feat, weight, h, w, num_embeds, h_im, w_im,
        value_offset,
        grad,
        grad_mc_ms_feat, grad_location_ptr, grad_weights_ptr
    );
}


__global__ void deformable_aggregation_grad_kernel_v1(
    const int64_t num_kernels,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    const float* grad_output,
    float* grad_mc_ms_feat,
    float* grad_sampling_location,
    float* grad_weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups,
    at::musa::FastDivmod fastdiv1,
    at::musa::FastDivmod fastdiv2,
    at::musa::FastDivmod fastdiv3,
    at::musa::FastDivmod fastdiv4,
    at::musa::FastDivmod fastdiv5,
    at::musa::FastDivmod fastdiv6

) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    const int weights_ptr = idx / (num_embeds / num_groups);

    uint32_t idx1, idx2, idx3, idx4, idx5, idx6;

    // at::musa::FastDivmod fastdv1((uint32_t)num_embeds);
    uint32_t channel_index; // uint32_t
    fastdiv1(idx1, channel_index, idx);

    // at::musa::FastDivmod fastdv2((uint32_t)num_scale);
    uint32_t scale_index; // uint32_t
    fastdiv2(idx2, scale_index, idx1);

    // at::musa::FastDivmod fastdv3((uint32_t)num_cams);
    uint32_t cam_index; // uint32_t
    fastdiv3(idx3, cam_index, idx2);

    // at::musa::FastDivmod fastdv4((uint32_t)num_pts);
    uint32_t pts_index; // uint32_t
    fastdiv4(idx4, pts_index, idx3);

    // at::musa::FastDivmod fastdv5((uint32_t)num_anchors);
    uint32_t anchor_index; // uint32_t
    fastdiv5(idx5, anchor_index, idx4);

    // at::musa::FastDivmod fastdv6((uint32_t)batch_size);
    uint32_t batch_index; // uint32_t
    fastdiv6(idx6, batch_index, idx5);

    // const int channel_index = idx % num_embeds;
    // idx /= num_embeds;
    // const int scale_index = idx % num_scale;
    // idx /= num_scale;

    // const int cam_index = idx % num_cams;
    // idx /= num_cams;
    // const int pts_index = idx % num_pts;
    // idx /= num_pts;

    // int anchor_index = idx % num_anchors;
    // idx /= num_anchors;
    // const int batch_index = idx % batch_size;
    // idx /= batch_size;

    anchor_index = batch_index * num_anchors + anchor_index;
    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1) return;
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1) return;
    
    const float grad = grad_output[anchor_index*num_embeds + channel_index];

    int cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    /* atomicAdd( */
    /*     output + anchor_index * num_embeds + channel_index, */
    /*     bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight */
    /* ); */
    const float weight = weights[weights_ptr];
    float *grad_weights_ptr = grad_weights + weights_ptr;
    float *grad_location_ptr = grad_sampling_location + loc_offset;
    bilinear_sampling_grad(
        mc_ms_feat, weight, h, w, num_embeds, h_im, w_im,
        value_offset,
        grad,
        grad_mc_ms_feat, grad_location_ptr, grad_weights_ptr
    );
}

__global__ void deformable_aggregation_grad_kernel_v2(
    const int64_t num_kernels,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    const float* grad_output,
    float* grad_mc_ms_feat,
    float* grad_sampling_location,
    float* grad_weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups,
    at::musa::FastDivmod fastdiv1,
    at::musa::FastDivmod fastdiv2,
    at::musa::FastDivmod fastdiv3,
    at::musa::FastDivmod fastdiv4,
    at::musa::FastDivmod fastdiv5,
    at::musa::FastDivmod fastdiv6

) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    const int weights_ptr = idx / (num_embeds / num_groups);

    uint32_t idx1, idx2, idx3, idx4, idx5, idx6;

    // at::musa::FastDivmod fastdv1((uint32_t)num_embeds);
    uint32_t channel_index; // uint32_t
    fastdiv1(idx1, channel_index, idx);

    // at::musa::FastDivmod fastdv2((uint32_t)num_scale);
    uint32_t scale_index; // uint32_t
    fastdiv2(idx2, scale_index, idx1);

    // at::musa::FastDivmod fastdv3((uint32_t)num_cams);
    uint32_t cam_index; // uint32_t
    fastdiv3(idx3, cam_index, idx2);

    // at::musa::FastDivmod fastdv4((uint32_t)num_pts);
    uint32_t pts_index; // uint32_t
    fastdiv4(idx4, pts_index, idx3);

    // at::musa::FastDivmod fastdv5((uint32_t)num_anchors);
    uint32_t anchor_index; // uint32_t
    fastdiv5(idx5, anchor_index, idx4);

    // at::musa::FastDivmod fastdv6((uint32_t)batch_size);
    uint32_t batch_index; // uint32_t
    fastdiv6(idx6, batch_index, idx5);

    // const int channel_index = idx % num_embeds;
    // idx /= num_embeds;
    // const int scale_index = idx % num_scale;
    // idx /= num_scale;

    // const int cam_index = idx % num_cams;
    // idx /= num_cams;
    // const int pts_index = idx % num_pts;
    // idx /= num_pts;

    // int anchor_index = idx % num_anchors;
    // idx /= num_anchors;
    // const int batch_index = idx % batch_size;
    // idx /= batch_size;

    anchor_index = batch_index * num_anchors + anchor_index;
    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;


    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1) return;
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1) return;

    const float grad = grad_output[anchor_index*num_embeds + channel_index];

    int cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    /* atomicAdd( */
    /*     output + anchor_index * num_embeds + channel_index, */
    /*     bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight */
    /* ); */
    const float weight = weights[weights_ptr];
    float *grad_weights_ptr = grad_weights + weights_ptr;
    float *grad_location_ptr = grad_sampling_location + loc_offset;
    bilinear_sampling_grad_v2(
        mc_ms_feat, weight, h, w, num_embeds, h_im, w_im,
        value_offset,
        grad,
        grad_mc_ms_feat, grad_location_ptr, grad_weights_ptr
    );
}

#define MAX_HEADS 8
__global__ void deformable_aggregation_grad_kernel_v3(
    const int64_t num_kernels,
    const float* mc_ms_feat_all,
    const int* spatial_shape_all,
    const int* scale_start_index_all,
    const float* sample_location_all,
    const float* weights_all,
    const float* grad_output_all,
    float* grad_mc_ms_feat_all,
    float* grad_sampling_location_all,
    float* grad_weights_all,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups,
    at::musa::FastDivmod fastdiv1,
    at::musa::FastDivmod fastdiv2,
    at::musa::FastDivmod fastdiv3,
    at::musa::FastDivmod fastdiv4,
    at::musa::FastDivmod fastdiv5,
    at::musa::FastDivmod fastdiv6

) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= num_kernels) return;

    // const int weights_ptr = idx / (num_embeds / num_groups);

    uint32_t idx1, idx2, idx3, idx4, idx5, idx6;

    // at::musa::FastDivmod fastdv1((uint32_t)num_embeds);
    uint32_t channel_index; // uint32_t
    // fastdiv1(idx1, channel_index, idx);

    // at::musa::FastDivmod fastdv2((uint32_t)num_scale);
    uint32_t scale_index; // uint32_t
    // fastdiv2(idx2, scale_index, idx);

    // at::musa::FastDivmod fastdv3((uint32_t)num_cams);
    uint32_t cam_index; // uint32_t
    fastdiv3(idx3, cam_index, idx);

    // at::musa::FastDivmod fastdv4((uint32_t)num_pts);
    uint32_t pts_index; // uint32_t
    fastdiv4(idx4, pts_index, idx3);

    // at::musa::FastDivmod fastdv5((uint32_t)num_anchors);
    uint32_t anchor_index; // uint32_t
    fastdiv5(idx5, anchor_index, idx4);

    // at::musa::FastDivmod fastdv6((uint32_t)batch_size);
    uint32_t batch_index; // uint32_t
    fastdiv6(idx6, batch_index, idx5);

    // sample_location index (no scale dim)
    int loc_flat_idx = ((batch_index * num_anchors + anchor_index) * num_pts + pts_index) * num_cams + cam_index;
    int loc_offset = loc_flat_idx << 1; // *2 for xy
    const float loc_w = sample_location_all[loc_offset];
    const float loc_h = sample_location_all[loc_offset + 1];

    // out early if sampling outside (match original)
    if (!(loc_w > 0.f && loc_w < 1.f && loc_h > 0.f && loc_h < 1.f)) {
        // To be safe: ensure we zero grad_weights for all scales/heads? 
        // Original kernel returned early too; host likely zero-initialized grad buffers.
        return;
    }

    // grad_output pointer for this (batch,anchor)
    const float *grad_output_ptr = grad_output_all + ((batch_index * num_anchors + anchor_index) * num_embeds);

    const int channels_per_head = num_embeds / num_groups;
    float grad_loc_x = 0.0f;
    float grad_loc_y = 0.0f;

    // per-scale loop: accumulate grad_weights per head for each scale, then write them
    for (int s = 0; s < num_scale; ++s) {
        // prepare per-head local accumulators (stack array, num_groups small)
        float grad_weights_local[MAX_HEADS];
        for (int h = 0; h < num_groups; ++h) grad_weights_local[h] = 0.0f;

        // compute spatial info for this (cam, scale)
        int cam_scale_index = cam_index * num_scale + s;
        const int h_img = spatial_shape_all[cam_scale_index * 2 + 0];
        const int w_img = spatial_shape_all[cam_scale_index * 2 + 1];

        const float h_im = loc_h * (float)h_img - 0.5f;
        const float w_im = loc_w * (float)w_img - 0.5f;
        const int h_low = floorf(h_im);
        const int w_low = floorf(w_im);
        const int h_high = h_low + 1;
        const int w_high = w_low + 1;
        const float lh = h_im - h_low, lw = w_im - w_low;
        const float hh = 1.0f - lh, hw = 1.0f - lw;
        const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

        // precompute offsets that are independent of channel
        const int w_stride = num_embeds;
        const int h_stride = w_img * w_stride;
        const int h_low_ptr_offset  = h_low * h_stride;
        const int h_high_ptr_offset = h_high * h_stride;
        const int w_low_ptr_offset  = w_low * w_stride;
        const int w_high_ptr_offset = w_high * w_stride;

        // base offset into mc_ms_feat for this batch & scale (without channel)
        const int value_base = (batch_index * num_feat + scale_start_index_all[cam_scale_index]) * num_embeds;

        // base index into weights (flattened): ... * num_scale + s, then * num_groups
        const int weights_base = (((((batch_index * num_anchors + anchor_index) * num_pts + pts_index)
                                     * num_cams + cam_index) * num_scale) + s) * num_groups;

        // CHANNEL loop: accumulate per-head grad_weights_local and sampling loc contributions
        for (int c = 0; c < num_embeds; ++c) {
            const int head = c / channels_per_head; // integer division

            const float grad_out_c = grad_output_ptr[c]; // grad_output for this channel
            // weight for this head at this scale
            const float weight_head = weights_all[weights_base + head];
            const float top_grad = grad_out_c * weight_head; // matches original top_grad_mc_ms_feat

            // compute per-channel ptrs (value_base already without channel)
            const int value_offset = value_base + c;
            const int ptr1 = (h_low >= 0 && w_low >= 0) ? (value_offset + h_low_ptr_offset + w_low_ptr_offset) : -1;
            const int ptr2 = (h_low >= 0 && w_high < w_img) ? (value_offset + h_low_ptr_offset + w_high_ptr_offset) : -1;
            const int ptr3 = (h_high < h_img && w_low >= 0) ? (value_offset + h_high_ptr_offset + w_low_ptr_offset) : -1;
            const int ptr4 = (h_high < h_img && w_high < w_img) ? (value_offset + h_high_ptr_offset + w_high_ptr_offset) : -1;

            const float v1 = (ptr1 != -1) ? mc_ms_feat_all[ptr1] : 0.0f;
            const float v2 = (ptr2 != -1) ? mc_ms_feat_all[ptr2] : 0.0f;
            const float v3 = (ptr3 != -1) ? mc_ms_feat_all[ptr3] : 0.0f;
            const float v4 = (ptr4 != -1) ? mc_ms_feat_all[ptr4] : 0.0f;

            // update grad_mc_ms_feat (still atomic)
            if (ptr1 != -1) atomicAdd(grad_mc_ms_feat_all + ptr1, w1 * top_grad);
            if (ptr2 != -1) atomicAdd(grad_mc_ms_feat_all + ptr2, w2 * top_grad);
            if (ptr3 != -1) atomicAdd(grad_mc_ms_feat_all + ptr3, w3 * top_grad);
            if (ptr4 != -1) atomicAdd(grad_mc_ms_feat_all + ptr4, w4 * top_grad);
            // if (ptr1 != -1) grad_mc_ms_feat_all[ptr1] += w1 * top_grad;
            // if (ptr2 != -1) grad_mc_ms_feat_all[ptr2] += w2 * top_grad;
            // if (ptr3 != -1) grad_mc_ms_feat_all[ptr3] += w3 * top_grad;
            // if (ptr4 != -1) grad_mc_ms_feat_all[ptr4] += w4 * top_grad;

            // grad_weights (per-head) increment: grad_out_c * val
            const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
            grad_weights_local[head] += grad_out_c * val;

            // grad_sampling_location contribution (per-channel): width * grad_w_weight_c * top_grad (and height * grad_h_weight_c * top_grad)
            float grad_h_weight_c = 0.0f, grad_w_weight_c = 0.0f;
            grad_h_weight_c -= (ptr1 != -1) ? hw * v1 : 0.0f;
            grad_h_weight_c -= (ptr2 != -1) ? lw * v2 : 0.0f;
            grad_h_weight_c += (ptr3 != -1) ? hw * v3 : 0.0f;
            grad_h_weight_c += (ptr4 != -1) ? lw * v4 : 0.0f;

            grad_w_weight_c -= (ptr1 != -1) ? hh * v1 : 0.0f;
            grad_w_weight_c += (ptr2 != -1) ? hh * v2 : 0.0f;
            grad_w_weight_c -= (ptr3 != -1) ? lh * v3 : 0.0f;
            grad_w_weight_c += (ptr4 != -1) ? lh * v4 : 0.0f;

            grad_loc_x += (float)w_img * grad_w_weight_c * top_grad; // x direction (width)
            grad_loc_y += (float)h_img * grad_h_weight_c * top_grad; // y direction (height)
            // if(idx==0 && s==3){
            //   printf("s:%d c:%d loc_offset:%d \n", s,c,loc_offset);
            // }
        } // end channel loop

        // write back per-head grad_weights for this scale (unique writer)
        const int grad_weights_base = (((((batch_index * num_anchors + anchor_index) * num_pts + pts_index)
                                         * num_cams + cam_index) * num_scale) + s) * num_groups;
        for (int h = 0; h < num_groups; ++h) {
            grad_weights_all[grad_weights_base + h] = grad_weights_local[h];
        }
    } // end scale loop

    // write sampling_location gradient once (unique writer per (b,a,p,cam))
    
    grad_sampling_location_all[loc_offset]     = grad_loc_x;
    grad_sampling_location_all[loc_offset + 1] = grad_loc_y;

}


__global__ void deformable_aggregation_grad_kernel_v4(
    // const int num_kernels,
    const float* mc_ms_feat_all,
    const int* spatial_shape_all,
    const int* scale_start_index_all,
    const float* sample_location_all,
    const float* weights_all,
    const float* grad_output_all,
    float* grad_mc_ms_feat_all,
    float* grad_sampling_location_all,
    float* grad_weights_all,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups,
    at::musa::FastDivmod fastdiv1,
    at::musa::FastDivmod fastdiv2,
    at::musa::FastDivmod fastdiv3,
    at::musa::FastDivmod fastdiv4,
    at::musa::FastDivmod fastdiv5,
    at::musa::FastDivmod fastdiv6

) {
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.x;

    const int total_blocks = batch_size * num_anchors * num_pts * num_cams;
    if (idx >= total_blocks) return;

    // const int weights_ptr = idx / (num_embeds / num_groups);

    uint32_t idx1, idx2, idx3, idx4, idx5, idx6;

    // at::musa::FastDivmod fastdv1((uint32_t)num_embeds);
    uint32_t channel_index; // uint32_t
    // fastdiv1(idx1, channel_index, idx);

    // at::musa::FastDivmod fastdv2((uint32_t)num_scale);
    uint32_t scale_index; // uint32_t
    // fastdiv2(idx2, scale_index, idx);

    // at::musa::FastDivmod fastdv3((uint32_t)num_cams);
    uint32_t cam_index; // uint32_t
    fastdiv3(idx3, cam_index, idx);

    // at::musa::FastDivmod fastdv4((uint32_t)num_pts);
    uint32_t pts_index; // uint32_t
    fastdiv4(idx4, pts_index, idx3);

    // at::musa::FastDivmod fastdv5((uint32_t)num_anchors);
    uint32_t anchor_index; // uint32_t
    fastdiv5(idx5, anchor_index, idx4);

    // at::musa::FastDivmod fastdv6((uint32_t)batch_size);
    uint32_t batch_index; // uint32_t
    fastdiv6(idx6, batch_index, idx5);

    const int tid = threadIdx.x;
    const int threads_needed = num_scale * num_embeds;
    if (tid >= threads_needed) {
        return;
    }
    fastdiv1(scale_index, channel_index, tid);
    const int channels_per_head = num_embeds / num_groups;
    const int head = channel_index / channels_per_head;

    // sample_location index (no scale dim)
    int loc_flat_idx = ((batch_index * num_anchors + anchor_index) * num_pts + pts_index) * num_cams + cam_index;
    int loc_offset = loc_flat_idx << 1; 
    const float loc_w = sample_location_all[loc_offset];
    const float loc_h = sample_location_all[loc_offset + 1];

    // out early if sampling outside (match original)
    if (!(loc_w > 0.f && loc_w < 1.f && loc_h > 0.f && loc_h < 1.f)) {
        return;
    }

    // per-scale cam_scale index
    const int cam_scale_index = cam_index * num_scale + scale_index;
    const int h_img = spatial_shape_all[cam_scale_index * 2 + 0];
    const int w_img = spatial_shape_all[cam_scale_index * 2 + 1];

    // base for mc_ms_feat (without channel)
    const int value_base = (batch_index * num_feat + scale_start_index_all[cam_scale_index]) * num_embeds;
    // base for weights: layout assumed [B, A, P, num_cams, num_scale, num_groups]
    const int weights_base = (((((batch_index * num_anchors + anchor_index) * num_pts + pts_index)
                                   * num_cams + cam_index) * num_scale) + scale_index) * num_groups;
    const float weight_head = weights_all[weights_base + head];

    // grad_output pointer for this (batch,anchor)
    const float *grad_output_ptr = grad_output_all + ((batch_index * num_anchors + anchor_index) * num_embeds);

    // prepare interpolation values
    const float h_im = loc_h * (float)h_img - 0.5f;
    const float w_im = loc_w * (float)w_img - 0.5f;
    const int h_low = floorf(h_im);
    const int w_low = floorf(w_im);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;
    const float lh = h_im - h_low, lw = w_im - w_low;
    const float hh = 1.0f - lh, hw = 1.0f - lw;
    const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;    

    // pointer offsets for mc_ms_feat
    const int w_stride = num_embeds;
    const int h_stride = w_img * w_stride;
    const int h_low_ptr_offset  = h_low  * h_stride;
    const int h_high_ptr_offset = h_high * h_stride;
    const int w_low_ptr_offset  = w_low  * w_stride;
    const int w_high_ptr_offset = w_high * w_stride;
    
    // === shared memory layout ===
    // shared size needed: num_scale * num_groups floats for grad_weights + 2 floats for grad_loc
    __shared__ float shmem[4*8+2]; // caller must supply proper shared_bytes
    // layout:
    // sh_weights [0 .. num_scale * num_groups - 1]   // index = scale_index * num_groups + head
    // sh_loc_x = shmem[num_scale*num_groups + 0]
    // sh_loc_y = shmem[num_scale*num_groups + 1]
    const int sh_w_offset = 0;
    const int sh_w_size = num_scale * num_groups;
    const int sh_loc_offset = sh_w_size; 

    for (int i = tid; i < sh_w_size + 2; i += blockDim.x) {
        shmem[i] = 0.0f;
    }
    __syncthreads();

    // === compute ptrs and values for this (scale, channel_index) ===
    const int value_offset = value_base + channel_index;
    const int ptr1 = (h_low >= 0 && w_low >= 0) ? (value_offset + h_low_ptr_offset + w_low_ptr_offset) : -1;
    const int ptr2 = (h_low >= 0 && w_high < w_img) ? (value_offset + h_low_ptr_offset + w_high_ptr_offset) : -1;
    const int ptr3 = (h_high < h_img && w_low >= 0) ? (value_offset + h_high_ptr_offset + w_low_ptr_offset) : -1;
    const int ptr4 = (h_high < h_img && w_high < w_img) ? (value_offset + h_high_ptr_offset + w_high_ptr_offset) : -1;

    const float v1 = (ptr1 != -1) ? mc_ms_feat_all[ptr1] : 0.0f;
    const float v2 = (ptr2 != -1) ? mc_ms_feat_all[ptr2] : 0.0f;
    const float v3 = (ptr3 != -1) ? mc_ms_feat_all[ptr3] : 0.0f;
    const float v4 = (ptr4 != -1) ? mc_ms_feat_all[ptr4] : 0.0f;

    // top_grad for this channel
    const float grad_out_c = grad_output_ptr[channel_index];
    const float top_grad = grad_out_c * weight_head; // matches original "top_grad_mc_ms_feat"

    // update global grad_mc_ms_feat (still global atomic)
    if (ptr1 != -1) atomicAdd(grad_mc_ms_feat_all + ptr1, w1 * top_grad);
    if (ptr2 != -1) atomicAdd(grad_mc_ms_feat_all + ptr2, w2 * top_grad);
    if (ptr3 != -1) atomicAdd(grad_mc_ms_feat_all + ptr3, w3 * top_grad);
    if (ptr4 != -1) atomicAdd(grad_mc_ms_feat_all + ptr4, w4 * top_grad);

    // compute contribution to grad_weights[scale, head]
    const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    const float gw_inc = grad_out_c * val;

    // atomicAdd into shmem
    const int sh_idx = sh_w_offset + scale_index * num_groups + head;
    atomicAdd(&shmem[sh_idx], gw_inc);

    // compute sampling location per-channel contributions
    float grad_h_weight_c = 0.0f, grad_w_weight_c = 0.0f;
    grad_h_weight_c -= (ptr1 != -1) ? hw * v1 : 0.0f;
    grad_h_weight_c -= (ptr2 != -1) ? lw * v2 : 0.0f;
    grad_h_weight_c += (ptr3 != -1) ? hw * v3 : 0.0f;
    grad_h_weight_c += (ptr4 != -1) ? lw * v4 : 0.0f;

    grad_w_weight_c -= (ptr1 != -1) ? hh * v1 : 0.0f;
    grad_w_weight_c += (ptr2 != -1) ? hh * v2 : 0.0f;
    grad_w_weight_c -= (ptr3 != -1) ? lh * v3 : 0.0f;
    grad_w_weight_c += (ptr4 != -1) ? lh * v4 : 0.0f;

    // NOTE: original used: atomicAdd(grad_sampling_location, width * grad_w_weight * top_grad_mc_ms_feat);
    // so each channel contributes width * grad_w_weight_c * top_grad
    const float loc_x_inc = (float)w_img * grad_w_weight_c * top_grad;
    const float loc_y_inc = (float)h_img * grad_h_weight_c * top_grad;

    // atomicAdd into shared loc
    atomicAdd(&shmem[sh_loc_offset + 0], loc_x_inc);
    atomicAdd(&shmem[sh_loc_offset + 1], loc_y_inc);

    __syncthreads();

    const int gw_base = (((((batch_index * num_anchors + anchor_index) * num_pts + pts_index)
                                 * num_cams + cam_index) * num_scale) ) * num_groups;

    if (tid < num_scale * num_groups) {
        int s = tid / num_groups;
        int h = tid & 7;
        int global_offset = gw_base + s * num_groups + h;
        grad_weights_all[global_offset] = shmem[sh_w_offset + s * num_groups + h];
    }

    // write grad_sampling_location
    if (tid == 0) {
        grad_sampling_location_all[loc_offset + 0] = shmem[sh_loc_offset + 0];
        grad_sampling_location_all[loc_offset + 1] = shmem[sh_loc_offset + 1];
    }

}


void deformable_aggregation(
    float* output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    const int64_t num_kernels = static_cast<int64_t>(batch_size) * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    const int64_t block_size = 128;
    const int64_t grid_size = (num_kernels + block_size - 1) / block_size;
    deformable_aggregation_kernel
        <<<grid_size, block_size>>>(
        num_kernels, output,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
    );
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        printf("[ERROR] Failed to launch deformable_aggregation_kernel: %d\n", err);
    }
}


void deformable_aggregation_v1(
    float* output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    at::musa::FastDivmod fastdiv1(num_embeds);
    at::musa::FastDivmod fastdiv2(num_scale);
    at::musa::FastDivmod fastdiv3(num_cams);
    at::musa::FastDivmod fastdiv4(num_pts);
    at::musa::FastDivmod fastdiv5(num_anchors);
    at::musa::FastDivmod fastdiv6(batch_size);

    const int64_t num_kernels = static_cast<int64_t>(batch_size) * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    const int64_t block_size = 128;
    const int64_t grid_size = (num_kernels + block_size - 1) / block_size;
    deformable_aggregation_kernel_v1
        <<<grid_size, block_size>>>(
        num_kernels, output,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups, 
        fastdiv1, fastdiv2, fastdiv3, fastdiv4, fastdiv5, fastdiv6
    );
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        printf("[ERROR] Failed to launch deformable_aggregation_kernel_v1: %d\n", err);
    }
}



void deformable_aggregation_grad(
  const float* mc_ms_feat,
  const int* spatial_shape,
  const int* scale_start_index,
  const float* sample_location,
  const float* weights,
  const float* grad_output,
  float* grad_mc_ms_feat,
  float* grad_sampling_location,
  float* grad_weights,
  int batch_size,
  int num_cams,
  int num_feat,
  int num_embeds,
  int num_scale,
  int num_anchors,
  int num_pts,
  int num_groups
) {
    const int64_t num_kernels = static_cast<int64_t>(batch_size) * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    const int64_t block_size = 128;
    const int64_t grid_size = (num_kernels + block_size - 1) / block_size;
    deformable_aggregation_grad_kernel
        <<<grid_size, block_size>>>(
        num_kernels,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
    );
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        printf("[ERROR] Failed to launch deformable_aggregation_grad_kernel: %d\n", err);
    }
}


void deformable_aggregation_grad_v1(
  const float* mc_ms_feat,
  const int* spatial_shape,
  const int* scale_start_index,
  const float* sample_location,
  const float* weights,
  const float* grad_output,
  float* grad_mc_ms_feat,
  float* grad_sampling_location,
  float* grad_weights,
  int batch_size,
  int num_cams,
  int num_feat,
  int num_embeds,
  int num_scale,
  int num_anchors,
  int num_pts,
  int num_groups
) {
    at::musa::FastDivmod fastdiv1(num_embeds);
    at::musa::FastDivmod fastdiv2(num_scale);
    at::musa::FastDivmod fastdiv3(num_cams);
    at::musa::FastDivmod fastdiv4(num_pts);
    at::musa::FastDivmod fastdiv5(num_anchors);
    at::musa::FastDivmod fastdiv6(batch_size);
    const int64_t num_kernels = static_cast<int64_t>(batch_size) * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    const int64_t block_size = 128;
    const int64_t grid_size = (num_kernels + block_size - 1) / block_size;
    deformable_aggregation_grad_kernel_v1
        <<<grid_size, block_size>>>(
        num_kernels,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups, fastdiv1, fastdiv2, fastdiv3, fastdiv4, fastdiv5, fastdiv6
    );
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        printf("[ERROR] Failed to launch deformable_aggregation_grad_kernel_v1: %d\n", err);
    }
}


void deformable_aggregation_grad_v2(
  const float* mc_ms_feat,
  const int* spatial_shape,
  const int* scale_start_index,
  const float* sample_location,
  const float* weights,
  const float* grad_output,
  float* grad_mc_ms_feat,
  float* grad_sampling_location,
  float* grad_weights,
  int batch_size,
  int num_cams,
  int num_feat,
  int num_embeds,
  int num_scale,
  int num_anchors,
  int num_pts,
  int num_groups
) {
    at::musa::FastDivmod fastdiv1(num_embeds);
    at::musa::FastDivmod fastdiv2(num_scale);
    at::musa::FastDivmod fastdiv3(num_cams);
    at::musa::FastDivmod fastdiv4(num_pts);
    at::musa::FastDivmod fastdiv5(num_anchors);
    at::musa::FastDivmod fastdiv6(batch_size);
    const int64_t num_kernels = static_cast<int64_t>(batch_size) * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    const int64_t block_size = 128;
    const int64_t grid_size = (num_kernels + block_size - 1) / block_size;
    deformable_aggregation_grad_kernel_v2
        // <<<8, 128>>>(
        <<<grid_size, block_size>>>(
        num_kernels,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups, fastdiv1, fastdiv2, fastdiv3, fastdiv4, fastdiv5, fastdiv6
    );
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        printf("[ERROR] Failed to launch deformable_aggregation_grad_kernel_v2: %d\n", err);
    }
}
   

void deformable_aggregation_grad_v3(
  const float* mc_ms_feat,
  const int* spatial_shape,
  const int* scale_start_index,
  const float* sample_location,
  const float* weights,
  const float* grad_output,
  float* grad_mc_ms_feat,
  float* grad_sampling_location,
  float* grad_weights,
  int batch_size,
  int num_cams,
  int num_feat,
  int num_embeds,
  int num_scale,
  int num_anchors,
  int num_pts,
  int num_groups
) {
    at::musa::FastDivmod fastdiv1(num_embeds);
    at::musa::FastDivmod fastdiv2(num_scale);
    at::musa::FastDivmod fastdiv3(num_cams);
    at::musa::FastDivmod fastdiv4(num_pts);
    at::musa::FastDivmod fastdiv5(num_anchors);
    at::musa::FastDivmod fastdiv6(batch_size);
    const long int num_kernels = batch_size * num_pts * num_anchors * num_cams;
    const int64_t block_size = 128;
    const int64_t grid_size = (num_kernels + block_size - 1) / block_size;
    deformable_aggregation_grad_kernel_v3
        <<<(int)ceil(((double)num_kernels/128)), 128>>>(
        // <<<1, 128>>>(
        num_kernels,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups, fastdiv1, fastdiv2, fastdiv3, fastdiv4, fastdiv5, fastdiv6
    );
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        printf("[ERROR] Failed to launch deformable_aggregation_grad_kernel_v3: %d\n", err);
    }
}


void deformable_aggregation_grad_v4(
  const float* mc_ms_feat,
  const int* spatial_shape,
  const int* scale_start_index,
  const float* sample_location,
  const float* weights,
  const float* grad_output,
  float* grad_mc_ms_feat,
  float* grad_sampling_location,
  float* grad_weights,
  int batch_size,
  int num_cams,
  int num_feat,
  int num_embeds,
  int num_scale,
  int num_anchors,
  int num_pts,
  int num_groups
) {
    at::musa::FastDivmod fastdiv1(num_embeds);
    at::musa::FastDivmod fastdiv2(num_scale);
    at::musa::FastDivmod fastdiv3(num_cams);
    at::musa::FastDivmod fastdiv4(num_pts);
    at::musa::FastDivmod fastdiv5(num_anchors);
    at::musa::FastDivmod fastdiv6(batch_size);
    const long int num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    deformable_aggregation_grad_kernel_v4
        <<<(int)ceil(((double)num_kernels/1024)), 1024>>>(
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups, fastdiv1, fastdiv2, fastdiv3, fastdiv4, fastdiv5, fastdiv6
    );
      
}