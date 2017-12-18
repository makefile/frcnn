#include <opencv2/highgui/highgui_c.h>
#include <stdint.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/core/version.hpp"
#if CV_MAJOR_VERSION == 2
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#elif CV_MAJOR_VERSION == 3
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#endif

#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/FRCNN/frcnn_vid_data_layer.hpp"

// caffe.proto > LayerParameter > FrcnnVidDataLayer
//   'source' field specifies the window_file
//   'crop_size' indicates the desired warped size
// label start from 1
// x1 y1 start from 1 , in the input file , so we -1 for every corrdinate

namespace caffe {

namespace Frcnn {

template <typename Dtype>
 FrcnnVidDataLayer<Dtype>::~FrcnnVidDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void FrcnnVidDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  // roi_data_file format
  // repeated:
  //   # image_index
  //   img_path (rel path)
  //   num_roi
  //   label x1 y1 x2 y2
  
  std::string default_config_file = this->layer_param_.window_data_param().config();
  FrcnnParam::load_param(default_config_file);
  FrcnnParam::print_param();

  // Did not 
  cache_images_ = this->layer_param_.window_data_param().cache_images();

  LOG(INFO) << "FrcnnVidDataLayer :" ;
  LOG(INFO) << "  source file :"
            << this->layer_param_.window_data_param().source() ; 
  LOG(INFO) << "  cache_images: "
            << ( cache_images_ ? "true" : "false" ) ; 
  LOG(INFO) << "  root_folder: "
            << this->layer_param_.window_data_param().root_folder() ;
  LOG(INFO) << "  Default Config File: "
            << default_config_file ;

  const std::string root_folder =
      this->layer_param_.window_data_param().root_folder();

  std::ifstream infile(this->layer_param_.window_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open roi_data file "
                       << this->layer_param_.window_data_param().source()
                       << std::endl;

  LOG(INFO) << "Initial rng seed";
  const unsigned int prefetch_rng_seed = FrcnnParam::rng_seed;
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  map<int, int> label_hist;
  label_hist.insert(std::make_pair(0, 0));
  this->vid_database_.clear();

  while (true) {
    VidPrepare<Dtype> vid_data;
    vid_data.init(PrefetchRand());
    bool ok = vid_data.load_data(infile);
    if (ok == false) break;
    vid_database_.push_back(vid_data);
    map<int, int> label_o = vid_data.count_label();
    for (map<int,int>::iterator it = label_o.begin(); it != label_o.end(); it++ ) {
      int label = it->first;
      int count = it->second;
      label_hist.insert(std::make_pair(0, 0));
      label_hist[label] += count;
    }
    if (vid_database_.size() % 1000 == 0) {
        LOG(INFO) << "num: " << vid_database_.size() << "processed";
    }
  }

  CHECK_GT(vid_database_.size(), 0) << "No Image In Ground Truth File";
  LOG(INFO) << "number of videos : " << vid_database_.size();

  for (map<int, int>::iterator it = label_hist.begin(); it != label_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " has " << label_hist[it->first] << " samples";
  }

  // image
  vector<float> scales = FrcnnParam::scales;
  max_short_ = *max_element(scales.begin(), scales.end());
  max_long_ = FrcnnParam::max_size;
  const int batch_size = 1;

  // data mean
  for (int i = 0; i < 3; i++) {
    mean_values_[i] = FrcnnParam::pixel_means[i];
  }

  // data image Input ..
  CHECK_GT(max_short_, 0);
  CHECK_GT(max_long_, 0);

  top[0]->Reshape(batch_size, 3, max_short_, max_long_);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(batch_size, 3, max_short_, max_long_);
  }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
            << top[0]->channels() << "," << top[0]->height() << ","
            << top[0]->width();

  // im_info: height width scale_factor
  top[1]->Reshape(1, 3, 1, 1);
  // gt_boxes: label x1 y1 x2 y2
  top[2]->Reshape(batch_size, 5, 1, 1);

  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(batch_size + 1, 5, 1, 1);
  }

} 

template <typename Dtype>
unsigned int FrcnnVidDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t *prefetch_rng =
      static_cast<caffe::rng_t *>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

// This function is called on prefetch thread
template <typename Dtype>
void FrcnnVidDataLayer<Dtype>::load_batch(Batch<Dtype> *batch) {
  // At each iteration, Give Batch images and
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;

  const vector<float> scales = FrcnnParam::scales;
  const bool mirror = FrcnnParam::use_flipped;
  const int batch_size = 1;

  timer.Start();

  // Select id for batch -> <0 if fliped
  int index = PrefetchRand() % vid_database_.size();

  bool do_mirror = mirror && PrefetchRand() % 2 && this->phase_ == TRAIN;
  float max_short = scales[PrefetchRand() % scales.size()];

  read_time += timer.MicroSeconds();

  // Prepare Image and labels;
  pair<vector<vector<float> >, string> X = vid_database_[index].Next();
  timer.Start();
  cv::Mat cv_img = cv::imread(X.second, CV_LOAD_IMAGE_COLOR);
  if (!cv_img.data) {
    LOG(FATAL) << "Could not open or find file " << X.second;
    return;
  }
  CHECK_EQ(cv_img.rows, vid_database_[index].H()) << "rows : " << cv_img.rows << ", Height : " << vid_database_[index].H();
  CHECK_EQ(cv_img.cols, vid_database_[index].W()) << "cols : " << cv_img.cols << ", Width  : " << vid_database_[index].W();
  cv::Mat src;
  cv_img.convertTo(src, CV_32FC3);
  if (do_mirror) {
    cv::flip(src, src, 1); // Flip
  }
  CHECK(src.isContinuous()) << "Warning : cv::Mat src is not Continuous !";
  CHECK_EQ(src.depth(), CV_32F) << "Image data type must be float 32 type";
  CHECK_EQ(src.channels(), 3) << "Image data type must be 3 channels";
  read_time += timer.MicroSeconds();

  timer.Start();
  // Convert by : sub means and resize
  // Image sub means
  for (int r = 0; r < src.rows; r++) {
    for (int c = 0; c < src.cols; c++) {
      int offset = (r * src.cols + c) * 3;
      reinterpret_cast<float *>(src.data)[offset + 0] -= this->mean_values_[0]; // B
      reinterpret_cast<float *>(src.data)[offset + 1] -= this->mean_values_[1]; // G
      reinterpret_cast<float *>(src.data)[offset + 2] -= this->mean_values_[2]; // R
    }
  }
  float im_scale = Frcnn::get_scale_factor(src.cols, src.rows, max_short, max_long_);
  cv::resize(src, src, cv::Size(), im_scale, im_scale);

  // resize data
  batch->data_.Reshape(batch_size, 3, src.rows, src.cols);
  Dtype *top_data = batch->data_.mutable_cpu_data();

  for (int r = 0; r < src.rows; r++) {
    for (int c = 0; c < src.cols; c++) {
      int cv_offset = (r * src.cols + c) * 3;
      int blob_shift = r * src.cols + c;
      top_data[0 * src.rows * src.cols + blob_shift] = reinterpret_cast<float *>(src.data)[cv_offset + 0];
      top_data[1 * src.rows * src.cols + blob_shift] = reinterpret_cast<float *>(src.data)[cv_offset + 1];
      top_data[2 * src.rows * src.cols + blob_shift] = reinterpret_cast<float *>(src.data)[cv_offset + 2];
    }
  }

  // label format:
  // labels x1 y1 x2 y2
  // special for frcnn , this first channel is -1 , width , height ,
  // width_with_pad , height_with_pad
  vector<vector<float> > rois = X.first;
  const int channels = rois.size() + 1;
  batch->label_.Reshape(channels, 5, 1, 1);
  Dtype *top_label = batch->label_.mutable_cpu_data();

  top_label[0] = src.rows; // height
  top_label[1] = src.cols; // width
  top_label[2] = im_scale; // im_scale: used to filter min size
  top_label[3] = 0;
  top_label[4] = 0;

  if (do_mirror) {
    for (int i = 0; i < rois.size(); i++) {
      CHECK_GE(rois[i][VidPrepare<Dtype>::X1], 0);
      if (rois[i][VidPrepare<Dtype>::Y2] >= cv_img.rows) {
        LOG(WARNING) << "rois[i][VidPrepare<Dtype>::Y2] : " << rois[i][VidPrepare<Dtype>::Y2] << ", cv_img.rows : " << cv_img.rows;
        rois[i][VidPrepare<Dtype>::Y2] = cv_img.rows - 1;
      }
      if (rois[i][VidPrepare<Dtype>::X2] >= cv_img.cols) {
        LOG(WARNING) << "rois[i][VidPrepare<Dtype>::X2] : " << rois[i][VidPrepare<Dtype>::X2] << ", cv_img.cols : " << cv_img.cols;
        rois[i][VidPrepare<Dtype>::X2] = cv_img.cols - 1;

      }
      float old_x1 = rois[i][VidPrepare<Dtype>::X1];
      float old_x2 = rois[i][VidPrepare<Dtype>::X2];
      rois[i][VidPrepare<Dtype>::X1] = cv_img.cols - old_x2 - 1; 
      rois[i][VidPrepare<Dtype>::X2] = cv_img.cols - old_x1 - 1; 
      CHECK_GE(rois[i][VidPrepare<Dtype>::X2], rois[i][VidPrepare<Dtype>::X1]);
      CHECK_GE(rois[i][VidPrepare<Dtype>::Y2], rois[i][VidPrepare<Dtype>::Y1]);
    }
  }
  CHECK_EQ(rois.size(), channels-1);
  for (int i = 1; i < channels; i++) {
    CHECK_EQ(rois[i-1].size(), VidPrepare<Dtype>::NUM);
    top_label[5 * i + 0] = rois[i-1][VidPrepare<Dtype>::X1] * im_scale; // x1
    top_label[5 * i + 1] = rois[i-1][VidPrepare<Dtype>::Y1] * im_scale; // y1
    top_label[5 * i + 2] = rois[i-1][VidPrepare<Dtype>::X2] * im_scale; // x2
    top_label[5 * i + 3] = rois[i-1][VidPrepare<Dtype>::Y2] * im_scale; // y2
    top_label[5 * i + 4] = rois[i-1][VidPrepare<Dtype>::LABEL];         // label

    if (top_label[5 * i + 0] < 0 ) {
      LOG(WARNING) << X.second << " : " << vid_database_[index].message() << " x1 : " << rois[i-1][VidPrepare<Dtype>::X1];
      top_label[5 * i + 0] = 0;
    }
    if (top_label[5 * i + 1] < 0 ) {
      LOG(WARNING) << X.second << " : " << vid_database_[index].message() << " y1 : " << rois[i-1][VidPrepare<Dtype>::Y1];
      top_label[5 * i + 1] = 0;
    }
    if (top_label[5 * i + 2] >= top_label[1] ) {
      LOG(WARNING) << X.second << " : " << vid_database_[index].message() << " x2 : " << rois[i-1][VidPrepare<Dtype>::X2];
      top_label[5 * i + 2] = top_label[1]-1;
    }
    if (top_label[5 * i + 3] >= top_label[0] ) {
      LOG(WARNING) << X.second << " : " << vid_database_[index].message() << " y2 : " << rois[i-1][VidPrepare<Dtype>::Y2];
      top_label[5 * i + 3] = top_label[0]-1;
    }
    
    CHECK_GE(top_label[5 * i + 0], 0 );
    CHECK_GE(top_label[5 * i + 1], 0 );
    CHECK_LE(top_label[5 * i + 2], top_label[1]) << mirror << " row : " << src.rows << ",  col : " << src.cols << ", im_scale : " 
            << im_scale << " | " << rois[i-1][VidPrepare<Dtype>::X2] << " , " << top_label[5 * i + 2];
    CHECK_LE(top_label[5 * i + 3], top_label[0]) << mirror << " row : " << src.rows << ",  col : " << src.cols << ", im_scale : " 
            << im_scale << " | " << rois[i-1][VidPrepare<Dtype>::Y2] << " , " << top_label[5 * i + 3];
    
  }

  trans_time += timer.MicroSeconds();

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
void FrcnnVidDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = this->prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(), top[0]->mutable_cpu_data());
  if (this->output_labels_) {
    caffe_copy(3, batch->label_.cpu_data(), top[1]->mutable_cpu_data());
    // Reshape to loaded labels.
    top[2]->Reshape(batch->label_.num()-1, batch->label_.channels(), batch->label_.height(), batch->label_.width());
    // Copy the labels.
    caffe_copy(batch->label_.count() - 5, batch->label_.cpu_data() + 5, top[2]->mutable_cpu_data());
  }
  this->prefetch_free_.push(batch);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(FrcnnVidDataLayer, Forward);
#endif

INSTANTIATE_CLASS(FrcnnVidDataLayer);
REGISTER_LAYER_CLASS(FrcnnVidData);

} // namespace Frcnn

} // namespace caffe
