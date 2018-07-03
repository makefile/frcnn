#include "roi_padding_layer.hpp"
#include "yaml-cpp/yaml.h"

namespace caffe {

namespace Frcnn {

template <typename Dtype>
void ROIPaddingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top) {
  YAML::Node params = YAML::Load(this->layer_param_.module_param().param_str());
  CHECK(params["pad_ratio"]) << "not found as parameter.";
  pad_ratio_ = params["pad_ratio"].as<float>();

  top[0]->Reshape(1, 5, 1, 1);
}

template <typename Dtype>
void ROIPaddingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
  const Dtype *rois_ = bottom[0]->cpu_data();
  top[0]->ReshapeLike(*bottom[0]);
  Dtype *top_rois_ = top[0]->mutable_cpu_data();
  int box_num = bottom[0]->num();
  for (int i = 0; i < box_num; i++) {
    // padding without consider boudary, since who use those rois will consider
    Dtype pad_w, pad_h;
    pad_w = (rois_[3] - rois_[1] + 1)*pad_ratio_;
    pad_h = (rois_[4] - rois_[2] + 1)*pad_ratio_;
    top_rois_[1] = rois_[1] - pad_w; // x1
    top_rois_[2] = rois_[2] - pad_h; // y1
    top_rois_[3] = rois_[3] + pad_w; // x2
    top_rois_[4] = rois_[4] + pad_h; // y2
    top_rois_[0] = rois_[0]; // whatever
    rois_ += bottom[0]->offset(1);
    top_rois_ += bottom[0]->offset(1);
  }
}


template <typename Dtype>
void ROIPaddingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    NOT_IMPLEMENTED;
  }
}

INSTANTIATE_CLASS(ROIPaddingLayer);
EXPORT_LAYER_MODULE_CLASS(ROIPadding);
//REGISTER_LAYER_CLASS(ROIPadding);

} // namespace frcnn

} // namespace caffe
