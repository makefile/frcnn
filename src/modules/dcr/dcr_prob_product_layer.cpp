// ------------------------------------------------------------------
// module DCR
// write by fyk
// ------------------------------------------------------------------

#include <cfloat>
#include <vector>

#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"
#include "dcr_prob_product_layer.hpp"
#include "yaml-cpp/yaml.h"

namespace caffe {
    
using namespace caffe::Frcnn;

template <typename Dtype>
void DCRProbProductLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void DCRProbProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {  
  // bottom: rcnn cls_prob, dcr top_index, dcr_cls_prob
  CHECK_EQ(bottom[1]->num(), bottom[2]->num());
  CHECK_LE(bottom[1]->num(), bottom[0]->num());
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void DCRProbProductLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Blob<Dtype>* cls_prob = bottom[0];
  Blob<Dtype>* top_index = bottom[1];
  Blob<Dtype>* dcr_cls_prob = bottom[2];
  const int rcnn_roi_num = cls_prob->num();
  const int cls_num = cls_prob->channels();
  const int dcr_roi_num = dcr_cls_prob->num();

  top[0]->ReshapeLike(*bottom[0]);
  Dtype* top_prob = top[0]->mutable_cpu_data();
  const Dtype* index_data = top_index->cpu_data();
  for (int i = 0; i < rcnn_roi_num; i++) {
    for (int c = 0; c < cls_num; c++) {
        int o_idx = i * cls_num + c;
        top_prob[o_idx] = cls_prob->cpu_data()[o_idx];
    }
  }
  for (int i = 0; i < dcr_roi_num; i++) {
    for (int c = 0; c < cls_num; c++) {
        int o_idx = index_data[i] * cls_num + c;
        top_prob[o_idx] *= dcr_cls_prob->cpu_data()[i * cls_num + c];
    }
  }
}

INSTANTIATE_CLASS(DCRProbProductLayer);
EXPORT_LAYER_MODULE_CLASS(DCRProbProduct);
//REGISTER_LAYER_CLASS(DCRProbProduct);

}  // namespace caffe
