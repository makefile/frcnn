// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

// modify by github.com/makefile
#include "fpn_utils.hpp"
#include "fpn_proposal_layer.m.hpp"
#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"  
#include "yaml-cpp/yaml.h"

namespace caffe {

namespace Frcnn {

using std::vector;

template <typename Dtype>
void FPNProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top) {

#ifndef CPU_ONLY
  CUDA_CHECK(cudaMalloc(&anchors_, sizeof(float) * FrcnnParam::anchors.size()));
  CUDA_CHECK(cudaMemcpy(anchors_, &(FrcnnParam::anchors[0]),
                        sizeof(float) * FrcnnParam::anchors.size(), cudaMemcpyHostToDevice));

  const int rpn_pre_nms_top_n = 
    this->phase_ == TRAIN ? FrcnnParam::rpn_pre_nms_top_n : FrcnnParam::test_rpn_pre_nms_top_n;
  CUDA_CHECK(cudaMalloc(&transform_bbox_, sizeof(float) * rpn_pre_nms_top_n * 4));
  CUDA_CHECK(cudaMalloc(&selected_flags_, sizeof(int) * rpn_pre_nms_top_n));

  const int rpn_post_nms_top_n = 
    this->phase_ == TRAIN ? FrcnnParam::rpn_post_nms_top_n : FrcnnParam::test_rpn_post_nms_top_n;
  CUDA_CHECK(cudaMalloc(&gpu_keep_indices_, sizeof(int) * rpn_post_nms_top_n));

#endif
  YAML::Node params = YAML::Load(this->layer_param_.module_param().param_str());
  CHECK(params["feat_strides"]) << "not found as parameter.";
  for (std::size_t i=0;i<params["feat_strides"].size();i++) {
    _feat_strides.push_back(params["feat_strides"][i].as<int>());
  }
  if (params["anchor_scales"]) {
    for (std::size_t i=0;i<params["anchor_scales"].size();i++) {
      _anchor_scales.push_back(params["anchor_scales"][i].as<int>());
    }
  } else {
    _anchor_scales.push_back(8);
    _anchor_scales.push_back(16);
    //_anchor_scales.push_back(32);//maybe too much
  }
  if (params["anchor_ratios"]) {
    for (std::size_t i=0;i<params["anchor_ratios"].size();i++) {
      _anchor_ratios.push_back(params["anchor_ratios"][i].as<float>());
    }
  } else {
    _anchor_ratios.push_back(0.5);
    _anchor_ratios.push_back(1);
    _anchor_ratios.push_back(2);
  }

  top[0]->Reshape(1, 5, 1, 1);
  /*
  if (top.size() > 1) {//fyk discard the score top
    top[1]->Reshape(1, 1, 1, 1);
  }*/
  if (this->phase_ == TEST) {
    top[1]->Reshape(1, 5, 1, 1);
    top[2]->Reshape(1, 5, 1, 1);
    top[3]->Reshape(1, 5, 1, 1);
    top[4]->Reshape(1, 5, 1, 1);
    //top[5]->Reshape(1, 5, 1, 1);
  }
}

template <typename Dtype>
void FPNProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
  DLOG(ERROR) << "========== enter proposal layer";
//  const Dtype *bottom_rpn_score = bottom[0]->cpu_data();  // rpn_cls_prob_reshape
//  const Dtype *bottom_rpn_bbox = bottom[1]->cpu_data();   // rpn_bbox_pred
// there are 5 rpn_score&rpn_bbox, and im_info is 11th bottom
  const Dtype *bottom_im_info = bottom[10]->cpu_data();    // im_info

  int rpn_pre_nms_top_n;
  int rpn_post_nms_top_n;
  float rpn_nms_thresh;
  int rpn_min_size;
  if (this->phase_ == TRAIN) {
    rpn_pre_nms_top_n = FrcnnParam::rpn_pre_nms_top_n;
    rpn_post_nms_top_n = FrcnnParam::rpn_post_nms_top_n;
    rpn_nms_thresh = FrcnnParam::rpn_nms_thresh;
    rpn_min_size = FrcnnParam::rpn_min_size;
  } else {
    rpn_pre_nms_top_n = FrcnnParam::test_rpn_pre_nms_top_n;
    rpn_post_nms_top_n = FrcnnParam::test_rpn_post_nms_top_n;
    rpn_nms_thresh = FrcnnParam::test_rpn_nms_thresh;
    rpn_min_size = FrcnnParam::test_rpn_min_size;
  }
  const float im_height = bottom_im_info[0];
  const float im_width = bottom_im_info[1];

  const Dtype bounds[4] = { im_width - 1, im_height - 1, im_width - 1, im_height -1 };
  const Dtype min_size = bottom_im_info[2] * rpn_min_size;
  //const int config_n_anchors = FrcnnParam::anchors.size() / 4;
  LOG_IF(ERROR, rpn_pre_nms_top_n <= 0 ) << "rpn_pre_nms_top_n : " << rpn_pre_nms_top_n;
  LOG_IF(ERROR, rpn_post_nms_top_n <= 0 ) << "rpn_post_nms_top_n : " << rpn_post_nms_top_n;
  if (rpn_pre_nms_top_n <= 0 || rpn_post_nms_top_n <= 0 ) return;

 //can be specified in proto or config file
 //from C2 to C5 (C6 is not used in Fast R-CNN for that almost none of rois is large to be assigned to)
 //const int _feat_strides[] = {4, 8, 16, 32, 64};//use it as base anchor size
 //const int inverse_anchor_sizes = {32, 64, 128, 256, 512};//not used
 //const int scales[] = {8, 16}; // for VOC
 //const int scales[] = {8}; // for COCO, only set one scale anchors in each pyramid feature
 //const int n_scales = sizeof(scales) / sizeof(int);
 //vector<int> anchor_scales(scales, scales + n_scales);
 //float arr[] = {0.5, 1, 2};
 //vector<float> anchor_ratios (arr, arr+3);
  const int config_n_anchors = _anchor_ratios.size() * _anchor_scales.size();

  std::vector<Point4f<Dtype> > anchors;
  typedef pair<Dtype, int> sort_pair;
  std::vector<sort_pair> sort_vector;

 //for loop
 for(int fp_i = 0; fp_i<_feat_strides.size(); fp_i++) {
  int score_blob_ind = fp_i * 2;
  int bbox_blob_ind  = fp_i * 2 + 1;
  const Dtype *bottom_rpn_score = bottom[score_blob_ind]->cpu_data();  // rpn_cls_prob_reshape
  const Dtype *bottom_rpn_bbox  = bottom[bbox_blob_ind]->cpu_data();   // rpn_bbox_pred
  const int num = bottom[bbox_blob_ind]->num();
  const int channes = bottom[bbox_blob_ind]->channels();
  const int height = bottom[bbox_blob_ind]->height();
  const int width = bottom[bbox_blob_ind]->width();
  CHECK(num == 1) << "only single item batches are supported";
  CHECK(channes % 4 == 0) << "rpn bbox pred channels should be divided by 4";

  DLOG(ERROR) << "========== generate anchors";
  vector<vector<int> > param_anchors = generate_anchors(_feat_strides[fp_i], _anchor_ratios, _anchor_scales);
  
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      for (int k = 0; k < config_n_anchors; k++) {
        Dtype score = bottom_rpn_score[config_n_anchors * height * width +
                                       k * height * width + j * width + i];
        //const int index = i * height * config_n_anchors + j * config_n_anchors + k;
        
        Point4f<Dtype> anchor(
           // FrcnnParam::anchors[k * 4 + 0] + i * FrcnnParam::feat_stride,  // shift_x[i][j];
           // FrcnnParam::anchors[k * 4 + 1] + j * FrcnnParam::feat_stride,  // shift_y[i][j];
           // FrcnnParam::anchors[k * 4 + 2] + i * FrcnnParam::feat_stride,  // shift_x[i][j];
           // FrcnnParam::anchors[k * 4 + 3] + j * FrcnnParam::feat_stride); // shift_y[i][j];
           param_anchors[k][0] + i * _feat_strides[fp_i],
           param_anchors[k][1] + j * _feat_strides[fp_i],
           param_anchors[k][2] + i * _feat_strides[fp_i],
           param_anchors[k][3] + j * _feat_strides[fp_i]);

        Point4f<Dtype> box_delta(
            bottom_rpn_bbox[(k * 4 + 0) * height * width + j * width + i],
            bottom_rpn_bbox[(k * 4 + 1) * height * width + j * width + i],
            bottom_rpn_bbox[(k * 4 + 2) * height * width + j * width + i],
            bottom_rpn_bbox[(k * 4 + 3) * height * width + j * width + i]);

        Point4f<Dtype> cbox = bbox_transform_inv(anchor, box_delta);
        
        // 2. clip predicted boxes to image
        for (int q = 0; q < 4; q++) {
          cbox.Point[q] = std::max(Dtype(0), std::min(cbox[q], bounds[q]));
        }
        // 3. remove predicted boxes with either height or width < threshold
        if((cbox[2] - cbox[0] + 1) >= min_size && (cbox[3] - cbox[1] + 1) >= min_size) {
          const int now_index = sort_vector.size();
          sort_vector.push_back(sort_pair(score, now_index)); 
          anchors.push_back(cbox);
        }
      }
    }
  }
 }//end loop FPN
  DLOG(ERROR) << "========== after clip and remove size < threshold box " << (int)sort_vector.size();

  std::sort(sort_vector.begin(), sort_vector.end(), std::greater<sort_pair>());
  const int n_anchors = std::min((int)sort_vector.size(), rpn_pre_nms_top_n);
  sort_vector.erase(sort_vector.begin() + n_anchors, sort_vector.end());
  //anchors.erase(anchors.begin() + n_anchors, anchors.end());
  std::vector<bool> select(n_anchors, true);

  // apply nms
  DLOG(ERROR) << "========== apply nms, pre nms number is : " << n_anchors;
  std::vector<Point4f<Dtype> > box_final;
  std::vector<Dtype> scores_;
  for (int i = 0; i < n_anchors && box_final.size() < rpn_post_nms_top_n; i++) {
    if (select[i]) {
      const int cur_i = sort_vector[i].second;
      for (int j = i + 1; j < n_anchors; j++)
        if (select[j]) {
          const int cur_j = sort_vector[j].second;
          if (get_iou(anchors[cur_i], anchors[cur_j]) > rpn_nms_thresh) {
            select[j] = false;
          }
        }
      box_final.push_back(anchors[cur_i]);
      scores_.push_back(sort_vector[i].first);
    }
  }

  DLOG(ERROR) << "rpn number after nms: " <<  box_final.size();

  DLOG(ERROR) << "========== copy to top";
  int n_level = 0; // fpn_levels
  if (this->phase_ == TEST) {
    n_level = 4;
    vector<vector<Point4f<Dtype> > > level_rois (n_level);
    vector<vector<Dtype> > level_scores (n_level, vector<Dtype>());
    for (size_t i = 0; i < box_final.size(); i++) {
      int level_idx = calc_level(box_final[i], n_level + 1) - 2;
      level_rois[level_idx].push_back(box_final[i]);
      level_scores[level_idx].push_back(scores_[i]);
    }
    // top_data[0] also need change
    box_final.clear();
    scores_.clear();
    for (size_t j = 0; j < n_level; j++) {
      if (level_rois[j].size() == 0) {
        level_rois[j].push_back(Point4f<Dtype>());
        box_final.push_back(Point4f<Dtype>());
        scores_.push_back(0);
      } else {
        box_final.insert(box_final.end(), level_rois[j].begin(), level_rois[j].end());
        scores_.insert(scores_.end(), level_scores[j].begin(), level_scores[j].end());
      }
    }
    split_top_rois_by_level(top,1,level_rois);
  }
  //train phase has proposal target layer,so there only output 1 total blob
  top[0]->Reshape(box_final.size(), 5, 1, 1);
  Dtype *top_data = top[0]->mutable_cpu_data();
  CHECK_EQ(box_final.size(), scores_.size());
  for (size_t i = 0; i < box_final.size(); i++) {
    Point4f<Dtype> &box = box_final[i];
    top_data[i * 5] = 0;
    for (int j = 1; j < 5; j++) {
      top_data[i * 5 + j] = box[j - 1];
    }
  }
  // optional score output
  if (top.size() > 1 + n_level) {
    top[1 + n_level]->Reshape(box_final.size(), 1, 1, 1);
    for (size_t i = 0; i < box_final.size(); i++) {
      top[1 + n_level]->mutable_cpu_data()[i] = scores_[i];
    }
  }

  DLOG(ERROR) << "========== exit proposal layer";
}

template <typename Dtype>
void FPNProposalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) {
      NOT_IMPLEMENTED;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FPNProposalLayer);
#endif

INSTANTIATE_CLASS(FPNProposalLayer);
//REGISTER_LAYER_CLASS(FPNProposal);
EXPORT_LAYER_MODULE_CLASS(FPNProposal);

} // namespace frcnn

} // namespace caffe
