// ------------------------------------------------------------------
// module layer of DCR
// write by fyk
// ------------------------------------------------------------------

#ifndef CAFFE_DCR_PROPOSAL_LAYER_HPP_
#define CAFFE_DCR_PROPOSAL_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class DCRProposalLayer : public Layer<Dtype> {
 public:
  explicit DCRProposalLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DCRProposal"; }
  
  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  //virtual inline int ExactNumTopBlobs() const { return 3; }
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 3; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /*virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);*/
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
  /*virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }*/
  
  Blob<Dtype> bbox_mean_;
  Blob<Dtype> bbox_std_;
  
  Blob<Dtype> bbox_pred_;

  float top_score_ratio_;
};

}  // namespace caffe

#endif  // CAFFE_DCR_PROPOSAL_LAYER_HPP_
