// --------------------------------------------------------
// Written by Bharat Singh, 2017.
// --------------------------------------------------------
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* a, double b) { return b; }
#endif
#include <algorithm>
#include <cfloat>
#include <vector>

#include "deformable_psroi_pooling_layer.hpp"
#include "caffe/util/gpu_util.cuh"

using std::max;
using std::min;

namespace caffe {
    template <typename DType>
    __device__ DType bilinear_interp(
      const DType* data,
      const DType x,
      const DType y,
      const int width,
      const int height) {
      int x1 = floor(x);
      int x2 = ceil(x);
      int y1 = floor(y);
      int y2 = ceil(y);
      DType dist_x = static_cast<DType>(x - x1);
      DType dist_y = static_cast<DType>(y - y1);
      DType value11 = data[y1*width + x1];
      DType value12 = data[y2*width + x1];
      DType value21 = data[y1*width + x2];
      DType value22 = data[y2*width + x2];
      DType value = (1 - dist_x)*(1 - dist_y)*value11 + (1 - dist_x)*dist_y*value12
        + dist_x*(1 - dist_y)*value21 + dist_x*dist_y*value22;
      return value;
    }

    template <typename DType>
    __global__ void DeformablePSROIPoolForwardKernel(
      const int count,
      const DType* bottom_data,
      const DType spatial_scale,
      const int channels,
      const int height, const int width,
      const int pooled_height, const int pooled_width,
      const DType* bottom_rois, const DType* bottom_trans,
      const bool no_trans,
      const DType trans_std,
      const int sample_per_part,
      const int output_dim,
      const int group_size,
      const int part_size,
      const int num_classes,
      const int channels_each_class,
      DType* top_data,
      DType* top_count) {
      CUDA_KERNEL_LOOP(index, count) {
        // The output is in order (n, ctop, ph, pw)
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int ctop = (index / pooled_width / pooled_height) % output_dim;
        int n = index / pooled_width / pooled_height / output_dim;

        // [start, end) interval for spatial sampling
        const DType* offset_bottom_rois = bottom_rois + n * 5;
        int roi_batch_ind = offset_bottom_rois[0];
        DType roi_start_w = static_cast<DType>(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
        DType roi_start_h = static_cast<DType>(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
        DType roi_end_w = static_cast<DType>(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
        DType roi_end_h = static_cast<DType>(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;

        // Force too small ROIs to be 1x1
        DType roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
        DType roi_height = max(roi_end_h - roi_start_h, 0.1);

        // Compute w and h at bottom
        DType bin_size_h = roi_height / static_cast<DType>(pooled_height);
        DType bin_size_w = roi_width / static_cast<DType>(pooled_width);

        DType sub_bin_size_h = bin_size_h / static_cast<DType>(sample_per_part);
        DType sub_bin_size_w = bin_size_w / static_cast<DType>(sample_per_part);

        int part_h = floor(static_cast<DType>(ph) / pooled_height*part_size);
        int part_w = floor(static_cast<DType>(pw) / pooled_width*part_size);
        int class_id = ctop / channels_each_class;
        DType trans_x = no_trans ? static_cast<DType>(0) :
          bottom_trans[(((n * num_classes + class_id) * 2) * part_size + part_h)*part_size + part_w] * trans_std;
        DType trans_y = no_trans ? static_cast<DType>(0) :
          bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size + part_h)*part_size + part_w] * trans_std;
        
        DType wstart = static_cast<DType>(pw)* bin_size_w
          + roi_start_w;
        wstart += trans_x * roi_width;
        DType hstart = static_cast<DType>(ph) * bin_size_h
          + roi_start_h;
        hstart += trans_y * roi_height;
        
        DType sum = 0;
        int count = 0;
        int gw = floor(static_cast<DType>(pw) * group_size / pooled_width);
        int gh = floor(static_cast<DType>(ph)* group_size / pooled_height);
        gw = min(max(gw, 0), group_size - 1);
        gh = min(max(gh, 0), group_size - 1);

        const DType* offset_bottom_data = bottom_data + (roi_batch_ind * channels) * height * width;
        for (int ih = 0; ih < sample_per_part; ih++) {
          for (int iw = 0; iw < sample_per_part; iw++) {
            DType w = wstart + iw*sub_bin_size_w;
            DType h = hstart + ih*sub_bin_size_h;
            // bilinear interpolation
            if (w<-0.5 || w>width - 0.5 || h<-0.5 || h>height - 0.5) {
              continue;
            }
            w = min(max(w, 0.), width - 1.);
            h = min(max(h, 0.), height - 1.);
            int c = (ctop*group_size + gh)*group_size + gw;
            DType val = bilinear_interp(offset_bottom_data + c*height*width, w, h, width, height);
            sum += val;
            count++;
          }
        }
        top_data[index] = count == 0 ? static_cast<DType>(0) : sum / count;
        top_count[index] = count;
      }
    }

  template <typename Dtype>
  void DeformablePSROIPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_rois = bottom[1]->gpu_data();
    const Dtype *bottom_trans = no_trans_ ? NULL : bottom[2]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    Dtype* mapping_channel_ptr = mapping_channel_.mutable_gpu_data();
    int count = top[0]->count();
    const int num_classes = no_trans_ ? 1 : bottom[2]->channels()/ 2;
    const int channels_each_class = no_trans_ ? output_dim_ : output_dim_ / num_classes;
    caffe_gpu_set(count, Dtype(0), top_data);
    caffe_gpu_set(count, Dtype(0), mapping_channel_ptr);
    // NOLINT_NEXT_LINE(whitespace/operators)
     DeformablePSROIPoolForwardKernel<Dtype> << <CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>> >(
        count, bottom_data, spatial_scale_, channels_, height_, width_, pooled_height_, pooled_width_,
        bottom_rois, bottom_trans, no_trans_, trans_std_, sample_per_part_, output_dim_, 
        group_size_, part_size_, num_classes, channels_each_class, top_data, mapping_channel_ptr);
    CUDA_POST_KERNEL_CHECK;
    
    //Forward_cpu(bottom, top);
  }
    template <typename DType>
    __global__ void DeformablePSROIPoolBackwardAccKernel(
      const int count,
      const DType* top_diff,
      const DType* top_count,
      const int num_rois,
      const DType spatial_scale,
      const int channels,
      const int height, const int width,
      const int pooled_height, const int pooled_width,
      const int output_dim,
      DType* bottom_data_diff, DType* bottom_trans_diff,
      const DType* bottom_data,
      const DType* bottom_rois,
      const DType* bottom_trans,
      const bool no_trans,
      const DType trans_std,
      const int sample_per_part,
      const int group_size,
      const int part_size,
      const int num_classes,
      const int channels_each_class) {
      CUDA_KERNEL_LOOP(index, count) {
        // The output is in order (n, ctop, ph, pw)
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int ctop = (index / pooled_width / pooled_height) % output_dim;
        int n = index / pooled_width / pooled_height / output_dim;

        // [start, end) interval for spatial sampling
        const DType* offset_bottom_rois = bottom_rois + n * 5;
        int roi_batch_ind = offset_bottom_rois[0];
        DType roi_start_w = static_cast<DType>(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
        DType roi_start_h = static_cast<DType>(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
        DType roi_end_w = static_cast<DType>(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
        DType roi_end_h = static_cast<DType>(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;

        // Force too small ROIs to be 1x1
        DType roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
        DType roi_height = max(roi_end_h - roi_start_h, 0.1);

        // Compute w and h at bottom
        DType bin_size_h = roi_height / static_cast<DType>(pooled_height);
        DType bin_size_w = roi_width / static_cast<DType>(pooled_width);

        DType sub_bin_size_h = bin_size_h / static_cast<DType>(sample_per_part);
        DType sub_bin_size_w = bin_size_w / static_cast<DType>(sample_per_part);

        int part_h = floor(static_cast<DType>(ph) / pooled_height*part_size);
        int part_w = floor(static_cast<DType>(pw) / pooled_width*part_size);
        int class_id = ctop / channels_each_class;
        DType trans_x = no_trans ? static_cast<DType>(0) :
          bottom_trans[(((n * num_classes + class_id) * 2) * part_size + part_h)*part_size + part_w] * trans_std;
        DType trans_y = no_trans ? static_cast<DType>(0) :
          bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size + part_h)*part_size + part_w] * trans_std;

        DType wstart = static_cast<DType>(pw)* bin_size_w
          + roi_start_w;
        wstart += trans_x * roi_width;
        DType hstart = static_cast<DType>(ph) * bin_size_h
          + roi_start_h;
        hstart += trans_y * roi_height;

        if (top_count[index] <= 0) {
          continue;
        }
        DType diff_val = top_diff[index] / top_count[index];
        const DType* offset_bottom_data = bottom_data + roi_batch_ind * channels * height * width;
        DType* offset_bottom_data_diff = bottom_data_diff + roi_batch_ind * channels * height * width;
        int gw = floor(static_cast<DType>(pw)* group_size / pooled_width);
        int gh = floor(static_cast<DType>(ph)* group_size / pooled_height);
        gw = min(max(gw, 0), group_size - 1);
        gh = min(max(gh, 0), group_size - 1);

        for (int ih = 0; ih < sample_per_part; ih++) {
          for (int iw = 0; iw < sample_per_part; iw++) {
            DType w = wstart + iw*sub_bin_size_w;
            DType h = hstart + ih*sub_bin_size_h;
            // bilinear interpolation
            if (w<-0.5 || w>width - 0.5 || h<-0.5 || h>height - 0.5) {
              continue;
            }
            w = min(max(w, 0.), width - 1.);
            h = min(max(h, 0.), height - 1.);
            int c = (ctop*group_size + gh)*group_size + gw;
            // backward on feature
            int x0 = floor(w);
            int x1 = ceil(w);
            int y0 = floor(h);
            int y1 = ceil(h);
            DType dist_x = w - x0, dist_y = h - y0;
            DType q00 = (1 - dist_x)*(1 - dist_y);
            DType q01 = (1 - dist_x)*dist_y;
            DType q10 = dist_x*(1 - dist_y);
            DType q11 = dist_x*dist_y;
            int bottom_index_base = c * height *width;
            atomicAdd(offset_bottom_data_diff + bottom_index_base + y0*width + x0, q00*diff_val);
            atomicAdd(offset_bottom_data_diff + bottom_index_base + y1*width + x0, q01*diff_val);
            atomicAdd(offset_bottom_data_diff + bottom_index_base + y0*width + x1, q10*diff_val);
            atomicAdd(offset_bottom_data_diff + bottom_index_base + y1*width + x1, q11*diff_val);

            if (no_trans) {
              continue;
            }
            DType U00 = offset_bottom_data[bottom_index_base + y0*width + x0];
            DType U01 = offset_bottom_data[bottom_index_base + y1*width + x0];
            DType U10 = offset_bottom_data[bottom_index_base + y0*width + x1];
            DType U11 = offset_bottom_data[bottom_index_base + y1*width + x1];
            DType diff_x = (U11*dist_y + U10*(1 - dist_y) - U01*dist_y - U00*(1 - dist_y))
              *trans_std*diff_val;
            diff_x *= roi_width;
            DType diff_y = (U11*dist_x + U01*(1 - dist_x) - U10*dist_x - U00*(1 - dist_x))
              *trans_std*diff_val;
            diff_y *= roi_height;

            atomicAdd(bottom_trans_diff + (((n * num_classes + class_id) * 2) * part_size + part_h)*part_size + part_w, diff_x);
            atomicAdd(bottom_trans_diff + (((n * num_classes + class_id) * 2 + 1)*part_size + part_h)*part_size + part_w, diff_y);
          }
        }
      }
    }

  template <typename Dtype>
  void DeformablePSROIPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
      return;
    }
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_rois = bottom[1]->gpu_data();
    
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype *bottom_trans = no_trans_ ? NULL : bottom[2]->gpu_data();
    Dtype* bottom_data_diff = bottom[0]->mutable_gpu_diff();
    Dtype *bottom_trans_diff = no_trans_ ? NULL : bottom[2]->mutable_gpu_diff();
    const int bottom_count = bottom[0]->count();
    const Dtype* mapping_channel_ptr = mapping_channel_.gpu_data();
    const int num_classes = no_trans_ ? 1 : bottom[2]->channels()/ 2;
    const int channels_each_class = no_trans_ ? output_dim_ : output_dim_ / num_classes;
    caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
    if(!no_trans_)
    	caffe_gpu_set(bottom[2]->count(), Dtype(0), bottom[2]->mutable_gpu_diff());
    caffe_gpu_set(bottom_count, Dtype(0), bottom_data_diff);
    const int count = top[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    DeformablePSROIPoolBackwardAccKernel<Dtype> << <CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>> >(
        count, top_diff, mapping_channel_ptr, top[0]->num(), spatial_scale_, channels_, height_, width_,
        pooled_height_, pooled_width_, output_dim_, bottom_data_diff, bottom_trans_diff,
        bottom_data, bottom_rois, bottom_trans, no_trans_, trans_std_, sample_per_part_,
        group_size_, part_size_, num_classes, channels_each_class);
    CUDA_POST_KERNEL_CHECK;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(DeformablePSROIPoolingLayer);

}  // namespace caffe
