// --------------------------------------------------------
// R-FCN
// Written by Afanti<afanti.deng@gmail.com>
// --------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>
#include <stdio.h>

#include "psroi_align_layer.hpp"
#include "caffe/util/gpu_util.cuh"

using std::max;
using std::min;

namespace caffe {

  template <typename Dtype>
        __device__ void bilinear_interpolate(const Dtype* bottom_data, const int height, const int width, Dtype h, Dtype w, Dtype & val) {

            // deal with cases that inverse elements are out of feature map boundary
            if (h < -0.5 || h > height - 0.5 || w < -0.5 || w > width - 0.5) return;

            if (h <= 0) h = 0;
            if (w <= 0) w = 0;

            
            int h_high;             // h_high 是比 h 大的最小整数
            int w_high;             // w_high 是比 w 大的最小整数
            int h_low = (int) h;    // h_low  是比 h 小的最大整数
            int w_low = (int) w;    // w_low  是比 w 小的最大整数

            if (w_low >= width - 1) {
                w_low = width - 1;
                w_high = width-1;
                w = (Dtype) w_low;
            } else 
                w_high = w_low + 1;

            if (h_low >= height - 1) {
                h_high = height-1;
                h_low = height - 1;
                h = (Dtype) h_low;
            } else 
                h_high = h_low + 1;

            
            Dtype l_dh = h - h_low;
            Dtype l_dw = w - w_low;
            Dtype h_dh = 1 - l_dh, h_dw = 1 - l_dw;

            // 进行双线性内插
            Dtype u1 = bottom_data[h_low * width + w_low];
            Dtype u2 = bottom_data[h_low * width + w_high];
            Dtype u3 = bottom_data[h_high * width + w_low];
            Dtype u4 = bottom_data[h_high * width + w_high];
            Dtype w1 = h_dh * h_dw, w2 = h_dh * l_dw, w3 = l_dh * h_dw, w4 = l_dh * l_dw;
            
            val = (w1 * u1 + w2 * u2 + w3 * u3 + w4 * u4);
        }

  template <typename Dtype>
  __global__ void PSROIAlignForward(
    const int nthreads,
    const Dtype* bottom_data,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois,
    const int output_dim,  // 输出通道数
    const int group_size,  // k*k*(c+1) 中的 k
    Dtype* top_data,
    int* mapping_channel,
    Dtype* sample_pos_data,
    const int sample_num) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int ctop = (index / pooled_width / pooled_height) % output_dim;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      Dtype roi_start_w =
        static_cast<Dtype>(bottom_rois[1]) * spatial_scale;
      Dtype roi_start_h =
        static_cast<Dtype>(bottom_rois[2]) * spatial_scale;
      Dtype roi_end_w =
        static_cast<Dtype>(bottom_rois[3] + 1.) * spatial_scale;
      Dtype roi_end_h =
        static_cast<Dtype>(bottom_rois[4] + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      // 获得当前RoI的宽和高在池化前特征图上的起始和结束索引值, 浮点数
      Dtype hstart = static_cast<Dtype>(ph) * bin_size_h;
      Dtype wstart = static_cast<Dtype>(pw) * bin_size_w;
      Dtype hend   = static_cast<Dtype>(ph + 1) * bin_size_h;
      Dtype wend   = static_cast<Dtype>(pw + 1) * bin_size_w;

      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart + roi_start_h, Dtype(0)), Dtype(height-1));
      hend = min(max(hend + roi_start_h, Dtype(0)), Dtype(height-1));
      wstart = min(max(wstart + roi_start_w, Dtype(0)), Dtype(width-1));
      wend = min(max(wend + roi_start_w, Dtype(0)), Dtype(width-1));
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      int gw = pw;
      int gh = ph;
      int c = (ctop*group_size + gh)*group_size + gw; // 

      // 在池化前特征图上采样点之间的距离，浮点数 (在 h 和 w 两个方向上)
      Dtype sample_h = bin_size_h / (sample_num + 1);
      Dtype sample_w = bin_size_w / (sample_num + 1);
      Dtype val = 0;
      bottom_data += (roi_batch_ind * channels + c) * height * width;
      Dtype out_sum = 0.0;
      int p_counter = -1;
      Dtype* sample_pos_diff = sample_pos_data + index * sample_num * sample_num * 2;
      for (int i = 1; i <= sample_num; ++i) {
          for (int j = 1; j <= sample_num; ++j) {
              ++p_counter;
              Dtype cur_h = hstart + i * sample_h;
              Dtype cur_w = wstart + j * sample_w;
              if (cur_h >= hend || cur_w >= wend) continue;
              bilinear_interpolate(bottom_data, height, width, cur_h, cur_w, val);
              out_sum += val; 
              sample_pos_diff[p_counter * 2 + 0] = cur_h;
              sample_pos_diff[p_counter * 2 + 1] = cur_w; 
              // updated = true;
          }
      }
      // Dtype bin_area = (hend - hstart) * (wend - wstart);
      top_data[index] = is_empty ? 0. : out_sum / static_cast<Dtype>(sample_num * sample_num);
      mapping_channel[index] = c;
    }
  }

  template <typename Dtype>
  void PSROIAlignLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_rois = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    Dtype* sample_pos_data = sample_pos_.mutable_gpu_data();
    int* mapping_channel_ptr = mapping_channel_.mutable_gpu_data();
    int count = top[0]->count();
    caffe_gpu_set(count, Dtype(0), top_data);
    caffe_gpu_set(count, -1, mapping_channel_ptr);
    // NOLINT_NEXT_LINE(whitespace/operators)
    PSROIAlignForward<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, bottom_data, spatial_scale_,
      channels_, height_, width_, pooled_height_,
      pooled_width_, bottom_rois, output_dim_, group_size_,
      top_data, mapping_channel_ptr, sample_pos_data, sample_num_);
    CUDA_POST_KERNEL_CHECK;
  }


  template <typename Dtype>
  __global__ void PSROIAlignBackwardAtomic(
    const int nthreads,
    const Dtype* top_diff,
    const int* mapping_channel,
    const Dtype* sample_pos_data,
    const int num_rois,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int output_dim,
    Dtype* bottom_diff,
    const Dtype* bottom_rois,
    const int sample_num) {
    // 遍历池化后特征图的每一个像素点
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int n = index / pooled_width / pooled_height / output_dim;
      
      // ------------------------------------ 计算当前 pooled 后的点在原图中的位置范围 ------------------------------------------------
      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      Dtype roi_start_w =
        static_cast<Dtype>(bottom_rois[1]) * spatial_scale;
      Dtype roi_start_h =
        static_cast<Dtype>(bottom_rois[2]) * spatial_scale;
      Dtype roi_end_w =
        static_cast<Dtype>(bottom_rois[3] + 1.) * spatial_scale;
      Dtype roi_end_h =
        static_cast<Dtype>(bottom_rois[4] + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      int hstart = floor(static_cast<Dtype>(ph) * bin_size_h
                          + roi_start_h);
      int wstart = floor(static_cast<Dtype>(pw)* bin_size_w
                          + roi_start_w);
      int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h
                        + roi_start_h);
      int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w
                        + roi_start_w);
      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0), width);
      wend = min(max(wend, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      // -------------------------------------------------------------------------------------

      // Compute c at bottom
      int c = mapping_channel[index];
      Dtype* offset_bottom_diff = bottom_diff +
        (roi_batch_ind * channels + c) * height * width;
      const Dtype* sample_pos_diff = sample_pos_data + index * sample_num * sample_num * 2;
      Dtype diff_val = is_empty ? 0. : top_diff[index] / (sample_num * sample_num);
      // diff_val = 0.;
      // diff_value = diff_val;
      // printf("diff_val: %f\n", float(diff_val));
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          for(int i = 0; i < sample_num * sample_num; ++i){
              Dtype d_h = abs(sample_pos_diff[2*i + 0] - h);
              Dtype d_w = abs(sample_pos_diff[2*i + 1] - w);
              if(d_h < 1 && d_w < 1){
                    int bottom_index = h*width + w;
                    caffe_gpu_atomic_add((1 - d_h)*(1 - d_w)*diff_val, offset_bottom_diff + bottom_index);
              }
          }
        }
      }
    }
  }

  template <typename Dtype>
  void PSROIAlignLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
      return;
    }

    const Dtype* bottom_rois = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int bottom_count = bottom[0]->count();
    const int* mapping_channel_ptr = mapping_channel_.gpu_data();
    caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
    caffe_gpu_set(bottom_count, Dtype(0), bottom_diff);
    const int count = top[0]->count();
    const Dtype* sample_pos_data = sample_pos_.gpu_data();
    // Dtype diff_value = static_cast<Dtype>(0);
    // NOLINT_NEXT_LINE(whitespace/operators)
    PSROIAlignBackwardAtomic<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, top_diff, mapping_channel_ptr, sample_pos_data,
      top[0]->num(), spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, output_dim_, bottom_diff,
      bottom_rois, sample_num_);
    // LOG(INFO) << "diff_value: " << diff_value;
    CUDA_POST_KERNEL_CHECK;
  }


  INSTANTIATE_LAYER_GPU_FUNCS(PSROIAlignLayer);

}  // namespace caffe
