// ------------------------------------------------------------------
// R-FCN
// Written by Yi Li
// ------------------------------------------------------------------

#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "psroi_pooling_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {
  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    PSROIPoolingParameter psroi_pooling_param =
      this->layer_param_.psroi_pooling_param();
    spatial_scale_ = psroi_pooling_param.spatial_scale();
    LOG(INFO) << "Spatial scale: " << spatial_scale_;

    CHECK_GT(psroi_pooling_param.output_dim(), 0)
      << "output_dim must be > 0";
    CHECK_GT(psroi_pooling_param.group_size(), 0)
      << "group_size must be > 0";

    output_dim_ = psroi_pooling_param.output_dim();
    group_size_ = psroi_pooling_param.group_size();
    pooled_height_ = group_size_;
    pooled_width_ = group_size_;
  }

  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    channels_ = bottom[0]->channels();
    CHECK_EQ(channels_, output_dim_*group_size_*group_size_)
      << "input channel number does not match layer parameters";
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    top[0]->Reshape(
      bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
    mapping_channel_.Reshape(
      bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
  }

  template <typename Dtype>
  static void PSROIPoolingForward(
    const int num,
    const Dtype* bottom_data,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois,
    const int output_dim,
    const int group_size,
    Dtype* top_data,
    int* mapping_channel) {
    for (int n = 0; n < num; ++n) {
      int roi_add = n*5;
      // [start, end) interval for spatial sampling
      int roi_batch_ind = bottom_rois[roi_add];
      Dtype roi_start_w =
        static_cast<Dtype>(round(bottom_rois[roi_add + 1])) * spatial_scale;
      Dtype roi_start_h =
        static_cast<Dtype>(round(bottom_rois[roi_add + 2])) * spatial_scale;
      Dtype roi_end_w =
        static_cast<Dtype>(round(bottom_rois[roi_add + 3]) + 1.) * spatial_scale;
      Dtype roi_end_h =
        static_cast<Dtype>(round(bottom_rois[roi_add + 4]) + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_width = max<Dtype>(roi_end_w - roi_start_w, 0.1);  // avoid 0
      Dtype roi_height = max<Dtype>(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      for (int ctop = 0; ctop < output_dim; ++ctop) {
        for (int ph = 0; ph < pooled_height; ++ph) {
          for (int pw = 0; pw < pooled_width; ++pw) {
            int index = n*output_dim*pooled_height*pooled_width + ctop*pooled_height*pooled_width + ph*pooled_width + pw;
            // The output is in order (n, ctop, ph, pw)

            int hstart = floor(static_cast<Dtype>(ph) * bin_size_h
                                + roi_start_h);
            int wstart = floor(static_cast<Dtype>(pw) * bin_size_w
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
     
            int gw = pw;
            int gh = ph;
            int c = (ctop*group_size + gh)*group_size + gw;

            //bottom_data += (roi_batch_ind * channels + c) * height * width;
            Dtype out_sum = 0;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                int bottom_index = h*width + w;
                out_sum += bottom_data[(roi_batch_ind * channels + c) * height * width + bottom_index];
              }
            }

            Dtype bin_area = (hend - hstart)*(wend - wstart);
            if (is_empty){
              top_data[index] = 0;
            }
            else{
              top_data[index] = out_sum/bin_area;
            }

            mapping_channel[index] = c;
          }
        }
      }
    }
  }

  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    //NOT_IMPLEMENTED;
    // fyk: cpu code from https://github.com/soeaver/py-RFCN-priv
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_rois = bottom[1]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    int* mapping_channel_ptr = mapping_channel_.mutable_cpu_data();
    int count = top[0]->count();
    caffe_set(count, Dtype(0), top_data);
    caffe_set(count, -1, mapping_channel_ptr);
    // NOLINT_NEXT_LINE(whitespace/operators)
    PSROIPoolingForward(bottom[1]->num(), bottom_data, spatial_scale_,
      channels_, height_, width_, pooled_height_,
      pooled_width_, bottom_rois, output_dim_, group_size_,
      top_data, mapping_channel_ptr);
  }

  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
#ifdef CPU_ONLY
  STUB_GPU(PSROIPoolingLayer);
#endif

  INSTANTIATE_CLASS(PSROIPoolingLayer);
  REGISTER_LAYER_CLASS(PSROIPooling);

}  // namespace caffe
