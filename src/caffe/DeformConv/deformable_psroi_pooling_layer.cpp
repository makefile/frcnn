// ------------------------------------------------------------------
// R-FCN
// Written by Yi Li
// ------------------------------------------------------------------

#include <cfloat>
#include <algorithm>

#include <string>
#include <utility>
#include <vector>

#include "deformable_psroi_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {
  template <typename Dtype>
  void DeformablePSROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    DeformablePSROIPoolingParameter deformable_psroi_pooling_param =
      this->layer_param_.deformable_psroi_pooling_param();
    spatial_scale_ = deformable_psroi_pooling_param.spatial_scale();
    LOG(INFO) << "Spatial scale: " << spatial_scale_;

    CHECK_GT(deformable_psroi_pooling_param.output_dim(), 0)
      << "output_dim must be > 0";
    CHECK_GT(deformable_psroi_pooling_param.group_size(), 0)
      << "group_size must be > 0";
    
    output_dim_ = deformable_psroi_pooling_param.output_dim();
    group_size_ = deformable_psroi_pooling_param.group_size();
    part_size_ = deformable_psroi_pooling_param.part_size();
    sample_per_part_ = deformable_psroi_pooling_param.sample_per_part();
    trans_std_ = deformable_psroi_pooling_param.trans_std();
    no_trans_ = deformable_psroi_pooling_param.no_trans();
    pooled_height_ = group_size_;
    pooled_width_ = group_size_;
  }

  template <typename Dtype>
  void DeformablePSROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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

#ifdef CPU_ONLY
  STUB_GPU(DeformablePSROIPoolingLayer);
#endif

  INSTANTIATE_CLASS(DeformablePSROIPoolingLayer);
  REGISTER_LAYER_CLASS(DeformablePSROIPooling);

}  // namespace caffe
