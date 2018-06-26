#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"
#include "caffe/common.hpp"

namespace caffe {

using namespace caffe::Frcnn;

std::vector<float> FrcnnParam::scales;
float FrcnnParam::max_size;
float FrcnnParam::batch_size;

float FrcnnParam::fg_fraction;
float FrcnnParam::fg_thresh;
// Overlap threshold for a ROI to be considered background (class = 0
// ifoverlap in [LO, HI))
float FrcnnParam::bg_thresh_hi;
float FrcnnParam::bg_thresh_lo;
bool FrcnnParam::use_flipped;
// fyk
int FrcnnParam::use_hist_equalize;
bool FrcnnParam::use_haze_free;
bool FrcnnParam::use_retinex;
float FrcnnParam::data_jitter;
float FrcnnParam::data_rand_scale;
bool FrcnnParam::data_rand_rotate;
float FrcnnParam::data_saturation;
float FrcnnParam::data_hue;
float FrcnnParam::data_exposure;

int FrcnnParam::im_size_align;
int FrcnnParam::roi_canonical_scale;
int FrcnnParam::roi_canonical_level;

int FrcnnParam::test_soft_nms; 
bool FrcnnParam::test_use_gpu_nms; 

// Train bounding-box regressors
bool FrcnnParam::bbox_reg; // Unuse
float FrcnnParam::bbox_thresh;
std::string FrcnnParam::snapshot_infix;
bool FrcnnParam::bbox_normalize_targets;
float FrcnnParam::bbox_inside_weights[4];
float FrcnnParam::bbox_normalize_means[4];
float FrcnnParam::bbox_normalize_stds[4];

// RPN to detect objects
float FrcnnParam::rpn_positive_overlap;
float FrcnnParam::rpn_negative_overlap;
// If an anchor statisfied by positive and negative conditions set to negative
bool FrcnnParam::rpn_clobber_positives;
float FrcnnParam::rpn_fg_fraction;
int FrcnnParam::rpn_batchsize;
float FrcnnParam::rpn_nms_thresh;
int FrcnnParam::rpn_pre_nms_top_n;
int FrcnnParam::rpn_post_nms_top_n;
// Proposal height and width both need to be greater than RPN_MIN_SIZE (at
// orig image scale)
float FrcnnParam::rpn_min_size;
// Deprecated (outside weights)
float FrcnnParam::rpn_bbox_inside_weights[4];
// Give the positive RPN examples weight of p * 1 / {num positives}
// and give negatives a weight of (1 - p)
// Set to -1.0 to use uniform example weighting
float FrcnnParam::rpn_positive_weight;
float FrcnnParam::rpn_allowed_border;

// ======================================== Test
std::vector<float> FrcnnParam::test_scales;
float FrcnnParam::test_max_size;
float FrcnnParam::test_nms;

bool FrcnnParam::test_bbox_reg;
// RPN to detect objects
float FrcnnParam::test_rpn_nms_thresh;
int FrcnnParam::test_rpn_pre_nms_top_n;
int FrcnnParam::test_rpn_post_nms_top_n;
// Proposal height and width both need to be greater than RPN_MIN_SIZE (at
// orig image scale)
float FrcnnParam::test_rpn_min_size;

// ========================================
// Means PIXEL
float FrcnnParam::pixel_means[3]; // BGR
int FrcnnParam::rng_seed;
float FrcnnParam::eps;
float FrcnnParam::inf;

// ======================================== 
int FrcnnParam::feat_stride;
std::vector<float> FrcnnParam::anchors;
float FrcnnParam::test_score_thresh;
float FrcnnParam::test_rpn_score_thresh;//fyk speed up for NMS
int FrcnnParam::n_classes;
int FrcnnParam::iter_test;

void FrcnnParam::load_param(const std::string default_config_path) {
  std::vector<float> v_tmp;

  str_map default_map = parse_json_config(default_config_path);

  FrcnnParam::scales = extract_vector("scales", default_map);
  FrcnnParam::max_size = extract_float("max_size", default_map);
  FrcnnParam::batch_size = extract_float("batch_size", default_map);

  FrcnnParam::fg_fraction = extract_float("fg_fraction", default_map);
  FrcnnParam::fg_thresh = extract_float("fg_thresh", default_map);
  FrcnnParam::bg_thresh_hi = extract_float("bg_thresh_hi", default_map);
  FrcnnParam::bg_thresh_lo = extract_float("bg_thresh_lo", default_map);
  FrcnnParam::use_flipped =
      static_cast<bool>(extract_int("use_flipped", default_map));
  // fyk: data enhancement & augmentation
  FrcnnParam::use_retinex =
      static_cast<bool>(extract_int("use_retinex", 0, default_map));
  FrcnnParam::use_haze_free =
      static_cast<bool>(extract_int("use_haze_free", 0, default_map));
  FrcnnParam::use_hist_equalize = extract_int("use_hist_equalize", 0, default_map);
  FrcnnParam::data_jitter = extract_float("data_jitter", -1, default_map);
  FrcnnParam::data_rand_scale = extract_float("data_rand_scale", 1, default_map);
  FrcnnParam::data_rand_rotate = static_cast<bool>(extract_int("data_rand_rotate", 0, default_map));
  FrcnnParam::data_hue = extract_float("data_hue", 0, default_map);
  FrcnnParam::data_saturation = extract_float("data_saturation", 0, default_map);
  FrcnnParam::data_exposure = extract_float("data_exposure", 0, default_map);

  FrcnnParam::im_size_align = extract_int("im_size_align", 1, default_map);
  FrcnnParam::roi_canonical_scale = extract_int("roi_canonical_scale", 224, default_map);
  FrcnnParam::roi_canonical_level = extract_int("roi_canonical_level", 4, default_map);
  FrcnnParam::test_soft_nms = extract_int("test_soft_nms", 0, default_map);
  FrcnnParam::test_use_gpu_nms = static_cast<bool>(extract_int("test_use_gpu_nms", 0, default_map));

  FrcnnParam::bbox_reg =
      static_cast<bool>(extract_int("bbox_reg", default_map));
  FrcnnParam::bbox_thresh = extract_float("bbox_thresh", default_map);
  FrcnnParam::snapshot_infix = extract_string("snapshot_infix", default_map);
  FrcnnParam::bbox_normalize_targets =
      static_cast<bool>(extract_int("bbox_normalize_targets", default_map));
  v_tmp = extract_vector("bbox_inside_weights", default_map);
  std::copy(v_tmp.begin(), v_tmp.end(), FrcnnParam::bbox_inside_weights);
  v_tmp = extract_vector("bbox_normalize_means", default_map);
  std::copy(v_tmp.begin(), v_tmp.end(), FrcnnParam::bbox_normalize_means);
  v_tmp = extract_vector("bbox_normalize_stds", default_map);
  std::copy(v_tmp.begin(), v_tmp.end(), FrcnnParam::bbox_normalize_stds);

  FrcnnParam::rpn_positive_overlap = extract_float("rpn_positive_overlap", default_map);
  FrcnnParam::rpn_negative_overlap = extract_float("rpn_negative_overlap", default_map);
  FrcnnParam::rpn_clobber_positives =
      static_cast<bool>(extract_int("rpn_clobber_positives", default_map));
  FrcnnParam::rpn_fg_fraction = extract_float("rpn_fg_fraction", default_map);
  FrcnnParam::rpn_batchsize = extract_int("rpn_batchsize", default_map);
  FrcnnParam::rpn_nms_thresh = extract_float("rpn_nms_thresh", default_map);
  FrcnnParam::rpn_pre_nms_top_n = extract_int("rpn_pre_nms_top_n", default_map);
  FrcnnParam::rpn_post_nms_top_n = extract_int("rpn_post_nms_top_n", default_map);
  FrcnnParam::rpn_min_size = extract_float("rpn_min_size", default_map);
  v_tmp = extract_vector("rpn_bbox_inside_weights", default_map);
  std::copy(v_tmp.begin(), v_tmp.end(), FrcnnParam::rpn_bbox_inside_weights);
  FrcnnParam::rpn_positive_weight = extract_float("rpn_positive_weight", default_map);
  FrcnnParam::rpn_allowed_border = extract_float("rpn_allowed_border", default_map);

  // ======================================== Test
  FrcnnParam::test_scales = extract_vector("test_scales", default_map);
  FrcnnParam::test_max_size = extract_float("test_max_size", default_map);
  FrcnnParam::test_nms = extract_float("test_nms", default_map);

  FrcnnParam::test_bbox_reg = static_cast<bool>(extract_float("test_bbox_reg", default_map));
  FrcnnParam::test_rpn_nms_thresh = extract_float("test_rpn_nms_thresh", default_map);
  FrcnnParam::test_rpn_pre_nms_top_n = extract_int("test_rpn_pre_nms_top_n", default_map);
  FrcnnParam::test_rpn_post_nms_top_n = extract_int("test_rpn_post_nms_top_n", default_map);
  FrcnnParam::test_rpn_min_size = extract_float("test_rpn_min_size", default_map);

  // ========================================
  v_tmp = extract_vector("pixel_means", default_map);
  std::copy(v_tmp.begin(), v_tmp.end(), FrcnnParam::pixel_means);
  FrcnnParam::rng_seed = extract_int("rng_seed", default_map);
  FrcnnParam::eps = extract_float("eps", default_map);
  FrcnnParam::inf = extract_float("inf", default_map);

  // ========================================
  FrcnnParam::feat_stride = extract_int("feat_stride", default_map);
  FrcnnParam::anchors = extract_vector("anchors", default_map);
  FrcnnParam::test_score_thresh = extract_float("test_score_thresh", default_map);
  FrcnnParam::test_rpn_score_thresh = extract_float("test_rpn_score_thresh", 0, default_map);
  FrcnnParam::n_classes = extract_int("n_classes", default_map);
  FrcnnParam::iter_test = extract_int("iter_test", default_map);
}

void FrcnnParam::print_param(){

  LOG(INFO) << "== Train  Parameters ==";
  LOG(INFO) << "scale             : " << float_to_string(FrcnnParam::scales); 
  LOG(INFO) << "max_size          : " << FrcnnParam::max_size;
  LOG(INFO) << "batch_size        : " << FrcnnParam::batch_size;

  LOG(INFO) << "fg_fraction       : " << FrcnnParam::fg_fraction;
  LOG(INFO) << "fg_thresh         : " << FrcnnParam::fg_thresh; 
  LOG(INFO) << "bg_thresh_hi      : " << FrcnnParam::bg_thresh_hi;
  LOG(INFO) << "bg_thresh_lo      : " << FrcnnParam::bg_thresh_lo;
  LOG(INFO) << "use_flipped       : " << (FrcnnParam::use_flipped ? "yes" : "no");

  LOG(INFO) << "use_bbox_reg      : " << (FrcnnParam::bbox_reg ? "yes" : "no");
  LOG(INFO) << "bbox_thresh       : " << FrcnnParam::bbox_thresh; 
  LOG(INFO) << "snapshot_infix    : " << FrcnnParam::snapshot_infix;
  LOG(INFO) << "normalize_targets : " << (FrcnnParam::bbox_normalize_targets ? "yes" : "no");

  LOG(INFO) << "rpn_pos_overlap   : " << FrcnnParam::rpn_positive_overlap; 
  LOG(INFO) << "rpn_neg_overlap   : " << FrcnnParam::rpn_negative_overlap; 
  LOG(INFO) << "clobber_positives : " << (FrcnnParam::rpn_clobber_positives ? "yes" : "no");
  LOG(INFO) << "rpn_fg_fraction   : " << FrcnnParam::rpn_fg_fraction;
  LOG(INFO) << "rpn_batchsize     : " << FrcnnParam::rpn_batchsize;
  LOG(INFO) << "rpn_nms_thresh    : " << FrcnnParam::rpn_nms_thresh;
  LOG(INFO) << "rpn_pre_nms_top_n : " << FrcnnParam::rpn_pre_nms_top_n;
  LOG(INFO) << "rpn_post_nms_top_n: " << FrcnnParam::rpn_post_nms_top_n; 
  LOG(INFO) << "rpn_min_size      : " << FrcnnParam::rpn_min_size;
  LOG(INFO) << "rpn_bbox_inside_weights :" << float_to_string(FrcnnParam::rpn_bbox_inside_weights);
  LOG(INFO) << "rpn_positive_weight     :" << FrcnnParam::rpn_positive_weight;
  LOG(INFO) << "rpn_allowed_border      :" << FrcnnParam::rpn_allowed_border;

  LOG(INFO) << "== Test   Parameters ==";
  LOG(INFO) << "test_scales          : " << float_to_string(FrcnnParam::test_scales); 
  LOG(INFO) << "test_max_size        : " << FrcnnParam::test_max_size; 
  LOG(INFO) << "test_nms             : " << FrcnnParam::test_nms; 
  LOG(INFO) << "test_bbox_reg        : " << (FrcnnParam::test_bbox_reg?"yes":"no");
  LOG(INFO) << "test_rpn_nms_thresh  : " << FrcnnParam::test_rpn_nms_thresh;
  LOG(INFO) << "rpn_pre_nms_top_n    : " << FrcnnParam::test_rpn_pre_nms_top_n;
  LOG(INFO) << "rpn_post_nms_top_n   : " << FrcnnParam::test_rpn_post_nms_top_n;
  LOG(INFO) << "test_rpn_min_sizen   : " << FrcnnParam::test_rpn_min_size; 

  LOG(INFO) << "== Global Parameters ==";
  LOG(INFO) << "pixel_means[BGR]     : " << FrcnnParam::pixel_means[0] <<  " , " << FrcnnParam::pixel_means[1] << " , " << FrcnnParam::pixel_means[2];
  LOG(INFO) << "rng_seed             : " << FrcnnParam::rng_seed;
  LOG(INFO) << "eps                  : " << FrcnnParam::eps; 
  LOG(INFO) << "inf                  : " << FrcnnParam::inf; 
  LOG(INFO) << "feat_stride          : " << FrcnnParam::feat_stride;
  LOG(INFO) << "anchors_size         : " << FrcnnParam::anchors.size();
  LOG(INFO) << "test_score_thresh    : " << FrcnnParam::test_score_thresh;
  LOG(INFO) << "n_classes            : " << FrcnnParam::n_classes;
  LOG(INFO) << "iter_test            : " << FrcnnParam::iter_test;
}

} // namespace detection
