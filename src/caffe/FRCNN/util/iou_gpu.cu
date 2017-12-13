// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
// ------------------------------------------------------------------

#include "caffe/FRCNN/util/frcnn_gpu_nms.hpp"
#include <vector>
#include <iostream>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

// fyk dev_boxes dim is 4 or 5
// boxes_dim must be a const value not a var,or NVCC will not compile
#define box_dim 4

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * box_dim];
  if (threadIdx.x < col_size) {
    for (int d = 0; d < box_dim; d ++) {
      block_boxes[threadIdx.x * box_dim + d] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * box_dim + d];
    }
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * box_dim;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * box_dim) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void _set_device(int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  CUDA_CHECK(cudaSetDevice(device_id));
}

namespace caffe {

namespace Frcnn {

// code from https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/nms_kernel.cu
// fyk:params is all cpu memory var, boxes_dim should be 4 instead of 5(x1,y1,x2,y2,confidence),the boxes_host should have already sorted by confidence from big to little
void _nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int device_id) {

  _set_device(device_id);

  float* boxes_dev = NULL;
  unsigned long long* mask_dev = NULL;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

  CUDA_CHECK(cudaMalloc(&boxes_dev,
                        boxes_num * boxes_dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(boxes_dev,
                        boxes_host,
                        boxes_num * boxes_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&mask_dev,
                        boxes_num * col_blocks * sizeof(unsigned long long)));

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  CUDA_CHECK(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  *num_out = num_to_keep;

  CUDA_CHECK(cudaFree(boxes_dev));
  CUDA_CHECK(cudaFree(mask_dev));
}

__global__ void overlaps_kernel(const int N, const int K, const float* dev_boxes,
                           const float * dev_query_boxes, float* dev_overlaps) {
  const int row_start = blockIdx.x;
  const int col_start = blockIdx.y;
  // note that this statement is different with nms_kernel, can also asign col_start with blockIdx.x and the order doesn't matter if the following statedments is right.

  // if (row_start > col_start) return;

  const int row_size =
        min(N - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(K - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 4];
  __shared__ float block_query_boxes[threadsPerBlock * 4];
  if (threadIdx.x < col_size) {
    block_query_boxes[threadIdx.x * 4 + 0] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
    block_query_boxes[threadIdx.x * 4 + 1] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
    block_query_boxes[threadIdx.x * 4 + 2] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
    block_query_boxes[threadIdx.x * 4 + 3] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
  }
  if (threadIdx.x < row_size) {
    block_boxes[threadIdx.x * 4 + 0] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 4 + 0];
    block_boxes[threadIdx.x * 4 + 1] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 4 + 1];
    block_boxes[threadIdx.x * 4 + 2] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 4 + 2];
    block_boxes[threadIdx.x * 4 + 3] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 4 + 3];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    for(int i = 0;i < col_size; i++) {
      int offset = row_start*threadsPerBlock * K + col_start*threadsPerBlock + threadIdx.x*K+ i ;
      dev_overlaps[offset] = devIoU(block_boxes + threadIdx.x * 4, block_query_boxes + i * 4);
    }
  }
}
// fyk:params is all cpu memory var, boxes_dim should be 4(x1,y1,x2,y2)
void _overlaps(float* overlaps,const float* boxes,const float* query_boxes, int n, int k, int device_id) {

  _set_device(device_id);

  float* overlaps_dev = NULL;
  float* boxes_dev = NULL;
  float* query_boxes_dev = NULL;

  CUDA_CHECK(cudaMalloc(&boxes_dev,
                        n * 4 * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(boxes_dev,
                        boxes,
                        n * 4 * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&query_boxes_dev,
                        k * 4 * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(query_boxes_dev,
                        query_boxes,
                        k * 4 * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&overlaps_dev,
                        n * k * sizeof(float)));

  dim3 blocks(DIVUP(n, threadsPerBlock),
              DIVUP(k, threadsPerBlock));
		
  dim3 threads(threadsPerBlock);

  overlaps_kernel<<<blocks, threads>>>(n, k,
                                    boxes_dev,
                                    query_boxes_dev,
                                    overlaps_dev);  

  CUDA_CHECK(cudaMemcpy(overlaps,
                        overlaps_dev,
                        n * k * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(overlaps_dev));
  CUDA_CHECK(cudaFree(boxes_dev));
  CUDA_CHECK(cudaFree(query_boxes_dev));

}

} // namespace frcnn

} // namespace caffe

