// ------------------------------------------------------------------
// makefile@github
// 2018/07/02
// ------------------------------------------------------------------
#ifndef CAFFE_ROI_PADDING_LAYER_HPP_
#define CAFFE_ROI_PADDING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

namespace Frcnn {

/*************************************************
ROI Global Context Operator enlarges the rois with its surrounding areas,
to provide contextual information

bottom: "rois"
top: "rois_ctx"
type: "ROIPadding"
roi_padding_param {
  pad_ratio: 0.5
}
**************************************************/
template <typename Dtype>
class ROIPaddingLayer : public Layer<Dtype> {
 public:
  explicit ROIPaddingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){};

  virtual inline const char* type() const { return "ROIPadding"; }

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    Dtype pad_ratio_;

};

}  // namespace frcnn

}  // namespace caffe

#endif // CAFFE_ROI_PADDING_LAYER_HPP_
