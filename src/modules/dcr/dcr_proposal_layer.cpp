// ------------------------------------------------------------------
// module DCR
// write by fyk
// ------------------------------------------------------------------

#include <cfloat>
#include <vector>

#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"
#include "dcr_proposal_layer.hpp"
#include "yaml-cpp/yaml.h"

namespace caffe {
    
using namespace caffe::Frcnn;

template <typename Dtype>
void DCRProposalLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  YAML::Node params = YAML::Load(this->layer_param_.module_param().param_str());
  CHECK(params["top_score_ratio"]) << "not found as parameter.";
  top_score_ratio_ = params["top_score_ratio"].as<float>();
  // bbox mean and std
  bbox_mean_.Reshape(4,1,1,1); bbox_std_.Reshape(4,1,1,1);
  for (int i = 0; i < 4; i++) {
    bbox_mean_.mutable_cpu_data()[i] = FrcnnParam::bbox_normalize_means[i];
    bbox_std_.mutable_cpu_data()[i] = FrcnnParam::bbox_normalize_stds[i];
    CHECK_GT(bbox_std_.mutable_cpu_data()[i],0);
  }
}

template <typename Dtype>
void DCRProposalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {  
  // bottom: bbox_pred, cls_score, rois
  CHECK_EQ(bottom[0]->num(),bottom[1]->num());
  CHECK_EQ(bottom[0]->num(),bottom[2]->num());
  /*if (bottom.size()>=3) {
    CHECK_EQ(bottom[0]->num(),bottom[2]->num());
    CHECK(this->phase_ == TRAIN);
  }*/
  //CHECK_EQ(bottom[0]->channels(),8); 
  CHECK_EQ(bottom[2]->channels(),5); 
  bbox_pred_.ReshapeLike(*bottom[2]);
}

template <typename Dtype>
void DCRProposalLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // notice that we can use boost::shared_ptr<Blob<Dtype> > rois(*bottom[2]); instead of rois(bottom[2]), the latter will double the memeory.
  Blob<Dtype>* bbox_pred = bottom[0];
  // currently only support class-aware,so we need cls_score
  Blob<Dtype>* cls_prob = bottom[1];
  Blob<Dtype>* rois = bottom[2];
  //LOG(INFO) << "shape bbox: " << bbox_pred->shape_string() << " cls_prob: " << cls_prob->shape_string() << " rois: " << rois->shape_string();
  const int roi_num = rois->num(); // bbox_pred is multi-class
  const int bbox_dim = rois->channels(); // [img_id x1 y1 x2 y2]
  const int cls_num = bbox_pred->channels() / 4;
  const Dtype* means = bbox_mean_.cpu_data();
  const Dtype* stds  = bbox_std_.cpu_data();
  Dtype* bbox_pred_data = bbox_pred_.mutable_cpu_data();

  // pred_scores
  typedef pair<Dtype, int> sort_pair;
  std::vector<sort_pair> sort_vector;

  for (int i = 0; i < roi_num; i++) {
    Point4f<Dtype> roi(rois->cpu_data()[(i * 5) + 1], rois->cpu_data()[(i * 5) + 2], rois->cpu_data()[(i * 5) + 3], rois->cpu_data()[(i * 5) + 4]);
    int cls_max = 1; // proposals should not include bg
    Dtype mx_score = cls_prob->cpu_data()[i * cls_num + cls_max];
    for (int c = 2; c < cls_num; c++) {
      Dtype score = cls_prob->cpu_data()[i * cls_num + c];
      if (score > mx_score) {
        mx_score = score;
        cls_max = c;
      }
    }
    
    sort_vector.push_back(sort_pair(mx_score, i));

    Point4f<Dtype> delta(bbox_pred->cpu_data()[(i * cls_num + cls_max) * 4 + 0] * stds[0] + means[0],
      bbox_pred->cpu_data()[(i * cls_num + cls_max) * 4 + 1] * stds[1] + means[1],
      bbox_pred->cpu_data()[(i * cls_num + cls_max) * 4 + 2] * stds[2] + means[2],
      bbox_pred->cpu_data()[(i * cls_num + cls_max) * 4 + 3] * stds[3] + means[3]);
    Point4f<Dtype> box = bbox_transform_inv(roi, delta);
    bbox_pred_data[i*bbox_dim + 0] = box[0];
    bbox_pred_data[i*bbox_dim + 1] = box[1];
    bbox_pred_data[i*bbox_dim + 2] = box[2]; 
    bbox_pred_data[i*bbox_dim + 3] = box[3];
  }
  
  std::sort(sort_vector.begin(), sort_vector.end(), std::greater<sort_pair>());
  int top_num = (int) (roi_num * top_score_ratio_);
  top_num = std::max(32, top_num); // gurantee at least

  vector<bool> valid_bbox_flags(roi_num,true);
  for (int idx = top_num; idx < roi_num; idx ++) {
    const int a_i = sort_vector[idx].second;
    valid_bbox_flags[a_i] = false;
  }
  // screen out mal-boxes
  if (this->phase_ == TRAIN) {
    for (int i = 0; i < roi_num; i++) {
      const int base_index = i*bbox_dim;
      if (bbox_pred_data[base_index] > bbox_pred_data[base_index+2] 
              || bbox_pred_data[base_index+1] > bbox_pred_data[base_index+3]) {
        valid_bbox_flags[i] = false;
      }
    }
  } 
  // screen out high IoU boxes, to remove redundant gt boxes
  /*if (bottom.size()==3 && this->phase_ == TRAIN) {
    const Dtype* match_gt_boxes = bottom[2]->cpu_data();
    const int gt_dim = bottom[2]->channels();
    const float gt_iou_thr = this->layer_param_.decode_bbox_param().gt_iou_thr();
    for (int i = 0; i < num; i++) {
      const float overlap = match_gt_boxes[i*gt_dim+gt_dim-1];
      if (overlap >= gt_iou_thr) {
        valid_bbox_flags[i] = false;
      }
    }
  }
  */
  vector<int> valid_bbox_ids;
  for (int i = 0; i < valid_bbox_flags.size(); i++) {
    if (valid_bbox_flags[i]) {
      valid_bbox_ids.push_back(i);
    }
  }
  const int keep_num = valid_bbox_ids.size();
  CHECK_GT(keep_num,0);
  
  top[0]->Reshape(keep_num, bbox_dim, 1, 1);
  Dtype* decoded_bbox_data = top[0]->mutable_cpu_data();
  top[1]->Reshape(keep_num, cls_num, 1, 1);
  Dtype* top_cls_prob = top[1]->mutable_cpu_data();
  Dtype* top_index = NULL;
  if (top.size() > 2) {
    // top_index is roi index of kept rois(top score & valid)
    top[2]->Reshape(keep_num, 1, 1, 1);
    top_index = top[2]->mutable_cpu_data();
    caffe_set(top[2]->count(), Dtype(0), top_index);
  }
  const Dtype* roi_data = rois->cpu_data();
  for (int i = 0; i < keep_num; i++) {
    const int keep_id = valid_bbox_ids[i];
    const int base_index = keep_id*bbox_dim;
    decoded_bbox_data[i*bbox_dim] =  roi_data[base_index];
    decoded_bbox_data[i*bbox_dim+1] = bbox_pred_data[base_index]; 
    decoded_bbox_data[i*bbox_dim+2] = bbox_pred_data[base_index+1]; 
    decoded_bbox_data[i*bbox_dim+3] = bbox_pred_data[base_index+2]; 
    decoded_bbox_data[i*bbox_dim+4] = bbox_pred_data[base_index+3];
    for (int c = 0; c < cls_num; c++) {
        top_cls_prob[i * cls_num + c] = cls_prob->cpu_data()[keep_id * cls_num + c];
    }
    if (top_index != NULL) top_index[i] = keep_id;
  }

}

INSTANTIATE_CLASS(DCRProposalLayer);
EXPORT_LAYER_MODULE_CLASS(DCRProposal);
//REGISTER_LAYER_CLASS(DCRProposal);

}  // namespace caffe
