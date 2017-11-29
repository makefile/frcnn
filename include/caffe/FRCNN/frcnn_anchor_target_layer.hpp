// ------------------------------------------------------------------
// Xuanyi . Refer to Dong Jian
// 2016/03/30
// ------------------------------------------------------------------
#ifndef CAFFE_FRCNN_ANCHOR_TARGET_LAYER_HPP_
#define CAFFE_FRCNN_ANCHOR_TARGET_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"

namespace caffe {

namespace Frcnn {

/*************************************************
FRCNN_ANCHOR_TARGET
Assign anchors to ground-truth targets. Produces anchor classification
labels and bounding-box regression targets.
bottom: 'rpn_cls_score'
bottom: 'gt_boxes'
bottom: 'im_info'
top: 'rpn_labels'
top: 'rpn_bbox_targets'
top: 'rpn_bbox_inside_weights'
top: 'rpn_bbox_outside_weights'
**************************************************/
template <typename Dtype>
class FrcnnAnchorTargetLayer : public Layer<Dtype> {
 public:
  explicit FrcnnAnchorTargetLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){};

  virtual inline const char* type() const { return "FrcnnAnchorTarget"; }

  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int MinTopBlobs() const { return 4; }
  virtual inline int MaxTopBlobs() const { return 4; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  vector<float> anchors_;
  int config_n_anchors_;
  int feat_stride_;
  float border_;

  Point4f<Dtype> _sum;
  Point4f<Dtype> _squared_sum;
  int _counts;
  int _fg_sum;
  int _bg_sum;
  int _count;
// For Debug
  inline void Info_Stds_Means_AvePos(const vector<Point4f<Dtype> > &targets, const vector<int> &labels){
//#ifdef DEBUG
#if 1
    CHECK_EQ(targets.size(), labels.size());
    const int n = targets.size();
    for (int index = 0; index < n; index++) {
      if (labels[index] == 1) {
        this->_counts ++;
        for (int j = 0; j < 4; j++) {
          this->_sum[j] = this->_sum[j] + targets[index][j];
          this->_squared_sum[j] = this->_squared_sum[j] + targets[index][j] * targets[index][j];
        }
      }
      _fg_sum += labels[index] == 1;
      _bg_sum += labels[index] == 0;
    }
    Point4f<Dtype> means, stds;
    for (int j = 0; j < 4; j++) if (this->_counts > 0 ) {
      means[j] = this->_sum[j] / this->_counts;
      stds[j] = sqrt(this->_squared_sum[j] / this->_counts - means[j]*means[j]);
    }
    LOG_EVERY_N(INFO, 1000) << "Info_Stds_Means_AvePos : COUNT : " << this->_counts;
    LOG_EVERY_N(INFO, 1000) << "STDS   : " << stds[0] << ", " << stds[1] << ", " << stds[2] << ", " << stds[3];
    LOG_EVERY_N(INFO, 1000) << "MEANS  : " << means[0] << ", " << means[1] << ", " << means[2] << ", " << means[3];
    this->_count++;
    LOG_EVERY_N(INFO, 1000) << "num_positive ave : " << float(_fg_sum) / _count;
    LOG_EVERY_N(INFO, 1000) << "num_negitive ave : " << float(_bg_sum) / _count;
#endif
  }
};

}  // namespace Frcnn

}  // namespace caffe

#endif  // CAFFE_FRCNN_ANCHOR_TARGET_LAYER_HPP_
