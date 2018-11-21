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
  // bottom: cls_prob, index
  CHECK_EQ(bottom[0]->num(),bottom[1]->num());
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void DCRProbProductLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Blob<Dtype>* cls_prob = bottom[0];
  Blob<Dtype>* top_index = bottom[1];
  Blob<Dtype>* dcr_cls_prob = bottom[2];
  const int roi_num = cls_prob->num();
  const int cls_num = cls_prob->channels();

  top[0]->ReshapeLike(*bottom[0]);
  Dtype* top_prob = top[0]->mutable_cpu_data();
  const Dtype* index_data = top_index->cpu_data();
  int dcr_idx = 0;
  for (int i = 0; i < roi_num; i++) {
    for (int c = 0; c < cls_num; c++) {
        int o_idx = i * cls_num + c;
        if (index_data[i] > 0) top_prob[o_idx] = cls_prob->cpu_data()[o_idx] * dcr_cls_prob->cpu_data()[dcr_idx * cls_num + c];
        else top_prob[o_idx] = cls_prob->cpu_data()[o_idx];
    }
    if (index_data[i] > 0) dcr_idx++;
  }

}

INSTANTIATE_CLASS(DCRProbProductLayer);
EXPORT_LAYER_MODULE_CLASS(DCRProbProduct);
//REGISTER_LAYER_CLASS(DCRProbProduct);

}  // namespace caffe
