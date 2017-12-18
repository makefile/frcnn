// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include "caffe/FRCNN/frcnn_proposal_layer.hpp"
#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"
#include "caffe/FRCNN/util/frcnn_gpu_nms.hpp"

#define USE_GPU_NMS // fyk: accelerate

namespace caffe {

namespace Frcnn {

using std::vector;

template <typename Dtype>
void FrcnnProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top) {
#ifdef GPU_CODE_HAS_NO_BUG //just not use GPU in this layer. remember to edit header file of this
//#ifndef CPU_ONLY
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
  top[0]->Reshape(1, 5, 1, 1);
  if (top.size() > 1) {
    top[1]->Reshape(1, 1, 1, 1);
  }
}

template <typename Dtype>
void FrcnnProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
  DLOG(ERROR) << "========== enter proposal layer";
  const Dtype *bottom_rpn_score = bottom[0]->cpu_data();  // rpn_cls_prob_reshape
  const Dtype *bottom_rpn_bbox = bottom[1]->cpu_data();   // rpn_bbox_pred
  const Dtype *bottom_im_info = bottom[2]->cpu_data();    // im_info

  const int num = bottom[1]->num();
  const int channes = bottom[1]->channels();
  const int height = bottom[1]->height();
  const int width = bottom[1]->width();
  CHECK(num == 1) << "only single item batches are supported";
  CHECK(channes % 5 == 0) << "rpn bbox pred channels should be divided by 5";//fyk change from 4 to 5 for adding angle

  const float im_height = bottom_im_info[0];
  const float im_width = bottom_im_info[1];

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
  const int config_n_anchors = FrcnnParam::anchors.size() / 3;
  LOG_IF(ERROR, rpn_pre_nms_top_n <= 0 ) << "rpn_pre_nms_top_n : " << rpn_pre_nms_top_n;
  LOG_IF(ERROR, rpn_post_nms_top_n <= 0 ) << "rpn_post_nms_top_n : " << rpn_post_nms_top_n;
  if (rpn_pre_nms_top_n <= 0 || rpn_post_nms_top_n <= 0 ) return;

  std::vector<Point5f<Dtype> > anchors;
  typedef pair<Dtype, int> sort_pair;
  std::vector<sort_pair> sort_vector;

  const Dtype bounds[4] = { im_width - 1, im_height - 1, im_width - 1, im_height -1 };
        Dtype pad_w = im_width  * 0.25;
        Dtype pad_h = im_height * 0.25;
        Dtype rbounds[4] = { - pad_w, - pad_h, im_width + pad_w, im_height + pad_h };
  const Dtype min_size = bottom_im_info[2] * rpn_min_size;

  DLOG(ERROR) << "========== generate anchors";
  
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      for (int k = 0; k < config_n_anchors; k++) {
        Dtype score = bottom_rpn_score[config_n_anchors * height * width +
                                       k * height * width + j * width + i];
        //const int index = i * height * config_n_anchors + j * config_n_anchors + k;

        Point5f<Dtype> anchor(
            i * FrcnnParam::feat_stride,  // shift_cx[i][j];
            j * FrcnnParam::feat_stride,  // shift_cy[i][j];
	    FrcnnParam::anchors[k * 3],   // w
	    FrcnnParam::anchors[k * 3 + 1],//h
	    FrcnnParam::anchors[k * 3 + 2]//theta
	    );
        Point5f<Dtype> box_delta(
            bottom_rpn_bbox[(k * 5 + 0) * height * width + j * width + i],
            bottom_rpn_bbox[(k * 5 + 1) * height * width + j * width + i],
            bottom_rpn_bbox[(k * 5 + 2) * height * width + j * width + i],
            bottom_rpn_bbox[(k * 5 + 3) * height * width + j * width + i],
            bottom_rpn_bbox[(k * 5 + 4) * height * width + j * width + i]);

        Point5f<Dtype> rbox = bbox_transform_inv(anchor, box_delta);
        if(rbox[2] <= 1 || rbox[3] <= 1) {
            // fyk: there are many predicted regressed boxes height < 1,and the GPU code of computing IoU does not consider this and will get wrong IoU,and the CPU code of OpenCV's rotatedRectangleIntersection is better but slower.
            //std::cout << "BAD: proposal rbbox w/h < 1" << std::endl;
            continue;
        }
	    Point4f<Dtype> cbox = rotate_outer_box_coordinates(rbox);
        // fyk: ignore some rbox which exceeds the image border too much, 25% padding space
        if(cbox[0] < rbounds[0] || cbox[1] < rbounds[1] || cbox[2] > rbounds[2] || cbox[3] > rbounds[3])
            continue;
        // let's ignore the clip on rotated box,and do on rectangle box
        // 2. clip predicted boxes to image
        for (int q = 0; q < 4; q++) {
          cbox.Point[q] = std::max(Dtype(0), std::min(cbox[q], bounds[q]));
        }
        // 3. remove predicted boxes with either height or width < threshold
        if((cbox[2] - cbox[0] + 1) >= min_size && (cbox[3] - cbox[1] + 1) >= min_size) {
          const int now_index = sort_vector.size();
          sort_vector.push_back(sort_pair(score, now_index)); 
          //anchors.push_back(cbox);
          anchors.push_back(rbox);
        }
      }
    }

  }

  DLOG(ERROR) << "========== after clip and remove size < threshold box " << (int)sort_vector.size();

  std::sort(sort_vector.begin(), sort_vector.end(), std::greater<sort_pair>());
  const int n_anchors = std::min((int)sort_vector.size(), rpn_pre_nms_top_n);
  sort_vector.erase(sort_vector.begin() + n_anchors, sort_vector.end());
  //anchors.erase(anchors.begin() + n_anchors, anchors.end());
  std::vector<bool> select(n_anchors, true);

  // apply nms
  DLOG(ERROR) << "========== apply nms, pre nms number is : " << n_anchors;
  std::vector<Point5f<Dtype> > box_final;
  std::vector<Dtype> scores_;
//fyk: use gpu
#ifdef USE_GPU_NMS
  std::vector<float> boxes_host(n_anchors * 5);
  for (int i=0; i<n_anchors; i++) {
    const int a_i = sort_vector[i].second;
    boxes_host[i * 5 + 0] = anchors[a_i][0];
    boxes_host[i * 5 + 1] = anchors[a_i][1];
    boxes_host[i * 5 + 2] = anchors[a_i][2];
    boxes_host[i * 5 + 3] = anchors[a_i][3];
    boxes_host[i * 5 + 4] = anchors[a_i][4];
  }
  int keep_out[n_anchors];//keeped index of boxes_host
  int num_out;//how many boxes are keeped
  // call gpu nms
  _rotate_nms(&keep_out[0], &num_out, &boxes_host[0], n_anchors, 5, rpn_nms_thresh);
  num_out = num_out < rpn_post_nms_top_n ? num_out : rpn_post_nms_top_n;
  for (int i=0; i<num_out; i++) {
    box_final.push_back(anchors[sort_vector[keep_out[i]].second]);
    scores_.push_back(sort_vector[keep_out[i]].first);
  }
#else
  for (int i = 0; i < n_anchors && box_final.size() < rpn_post_nms_top_n; i++) {
    if (select[i]) {
      const int cur_i = sort_vector[i].second;
      for (int j = i + 1; j < n_anchors; j++)
        if (select[j]) {
          const int cur_j = sort_vector[j].second;
          Dtype iou = skew_iou(anchors[cur_i], anchors[cur_j]);
          if (iou > rpn_nms_thresh) {
            select[j] = false;
          }
          else if(iou > 0.3) {//fyk consider angles.if iou between [0.3,0.7], remove the proposal with angle difference less than π/12)
          	if( fabs(anchors[cur_i][4] - anchors[cur_j][4]) < M_PI/12 ) select[j] = false;
          }
        }
      box_final.push_back(anchors[cur_i]);
      scores_.push_back(sort_vector[i].first);
    }
  }
#endif
  DLOG(ERROR) << "rpn number after nms: " <<  box_final.size();

  DLOG(ERROR) << "========== copy to top";
  top[0]->Reshape(box_final.size(), 6, 1, 1);
  Dtype *top_data = top[0]->mutable_cpu_data();
  CHECK_EQ(box_final.size(), scores_.size());
  for (size_t i = 0; i < box_final.size(); i++) {
    Point5f<Dtype> &box = box_final[i];
    top_data[i * 6] = 0;//fyk set this 0 to indicate single item.
    for (int j = 1; j < 6; j++) {
      top_data[i * 6 + j] = box[j - 1];
    }
  }

  if (top.size() > 1) { // fyk can also output scores of proposals
    top[1]->Reshape(box_final.size(), 1, 1, 1);
    for (size_t i = 0; i < box_final.size(); i++) {
      top[1]->mutable_cpu_data()[i] = scores_[i];
    }
  }

  DLOG(ERROR) << "========== exit proposal layer";
}

template <typename Dtype>
void FrcnnProposalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) {
      NOT_IMPLEMENTED;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FrcnnProposalLayer);
#endif

INSTANTIATE_CLASS(FrcnnProposalLayer);
REGISTER_LAYER_CLASS(FrcnnProposal);

} // namespace frcnn

} // namespace caffe
