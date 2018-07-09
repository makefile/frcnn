#include <opencv2/highgui/highgui_c.h>
#include <stdint.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
//#include "opencv2/core/version.hpp"
//#if CV_MAJOR_VERSION == 2
//#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//
//#elif CV_MAJOR_VERSION == 3
//#include "opencv2/core.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"
//
//#endif
//fyk
#include "util/equalize_hist.hpp"

#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/FRCNN/frcnn_roi_data_layer.hpp"
#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"

// caffe.proto > LayerParameter > FrcnnRoiDataLayer
//   'source' field specifies the window_file
//   'crop_size' indicates the desired warped size
// label start from 1
// x1 y1 start from 1 , in the input file , so we -1 for every corrdinate

namespace caffe {

namespace Frcnn {

template <typename Dtype>
 FrcnnRoiDataLayer<Dtype>::~FrcnnRoiDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void FrcnnRoiDataLayer<Dtype>::DataLayerSetUp(
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
  cache_images_ = this->layer_param_.window_data_param().cache_images();

  LOG(INFO) << "FrcnnRoiDataLayer :" ;
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

  map<int, int> label_hist;
  label_hist.insert(std::make_pair(0, 0));
  roi_database_.clear();

  DataPrepare data_load;
  while( data_load.load_WithDiff(infile) ) {
    string image_path = data_load.GetImagePath(root_folder);
    //int image_index = data_load.GetImageIndex();
    image_database_.push_back(image_path);
    lines_.push_back(image_database_.size()-1);
    if (cache_images_) {
      Datum datum;
      if (!ReadFileToDatum(image_path, &datum)) {
        LOG(ERROR) << "Could not open or find file " << image_path;
        return;
      }
      image_database_cache_.push_back(std::make_pair(image_path, datum));
    }
    vector<vector<float> > rois = data_load.GetRois( false );
    for (size_t i = 0; i < rois.size(); ++i) {
      int label = rois[i][DataPrepare::LABEL];
      label_hist.insert(std::make_pair(label, 0));
      label_hist[label]++;
    }
    roi_database_.push_back(rois);
    if (lines_.size() % 1000 == 0) {
        LOG(INFO) << "num: " << lines_.size() << " " << image_path << " "
            << "rois to process: " << rois.size();
    }
  }

  CHECK_GT(lines_.size(), 0) << "No Image In Ground Truth File";
  LOG(INFO) << "number of images: " << lines_.size();

  for (map<int, int>::iterator it = label_hist.begin(); it != label_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " has " << label_hist[it->first] << " samples";
  }

  // image
  vector<float> scales = FrcnnParam::scales;
  max_short_ = *max_element(scales.begin(), scales.end());
  max_long_ = FrcnnParam::max_size;
  //const int batch_size = 1;
  // fyk modify for supporting batch > 1
  const int batch_size = FrcnnParam::IMS_PER_BATCH;

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

  // im_info: height width scale_factor, fyk add 'box_num',and an additional padding not used for now.
  // fyk modify for supporting batch > 1
  top[1]->Reshape(batch_size, 5, 1, 1);
  // gt_boxes: label x1 y1 x2 y2
  top[2]->Reshape(batch_size, 5, 1, 1);//usally at least 1, at setup this reshape num is not used.
LOG(INFO) << "prefetch_.size(): " << this->prefetch_.size();//will prefetch this number of batches in data thread.
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(batch_size, 5, 1, 1);//can be any number
  }

  LOG(INFO) << "Shuffling data";
  const unsigned int prefetch_rng_seed = FrcnnParam::rng_seed;
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  lines_id_ = 0; // First Shuffle
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
} 

template <typename Dtype>
void FrcnnRoiDataLayer<Dtype>::ShuffleImages() { //fyk: proceed the next batch
  //lines_id_++;
  // fyk modify for supporting batch > 1
  lines_id_ += FrcnnParam::IMS_PER_BATCH;
  if (lines_id_ >= lines_.size()) {
    // We have reached the end. Restart from the first.
    DLOG(INFO) << "Restarting data prefetching from start.";
    lines_id_ = 0;
    CHECK(prefetch_rng_);
    caffe::rng_t* prefetch_rng =
        static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    shuffle(lines_.begin(), lines_.end(), prefetch_rng);
  }
}

template <typename Dtype>
unsigned int FrcnnRoiDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t *prefetch_rng =
      static_cast<caffe::rng_t *>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
void FrcnnRoiDataLayer<Dtype>::CheckResetRois(vector<vector<float> > &rois, const string image_path, const float cols, const float rows, const float im_scale) {
  for (int i = 0; i < rois.size(); i++) {
    bool ok = rois[i][DataPrepare::X1] >= 0 && rois[i][DataPrepare::Y1] >= 0 && 
        rois[i][DataPrepare::X2] < cols && rois[i][DataPrepare::Y2] < rows;
    if (ok == false) {
      DLOG(INFO) << "Roi Data Check Failed : " << image_path << " [" << i << "]";
      DLOG(INFO) << " row : " << rows << ",  col : " << cols << ", im_scale : " << im_scale << " | " << rois[i][DataPrepare::X1] << ", " << rois[i][DataPrepare::Y1] << ", " << rois[i][DataPrepare::X2] << ", " << rois[i][DataPrepare::Y2];
      rois[i][DataPrepare::X1] = std::max(0.f, rois[i][DataPrepare::X1]);
      rois[i][DataPrepare::Y1] = std::max(0.f, rois[i][DataPrepare::Y1]);
      rois[i][DataPrepare::X2] = std::min(cols-1, rois[i][DataPrepare::X2]);
      rois[i][DataPrepare::Y2] = std::min(rows-1, rois[i][DataPrepare::Y2]);
    }
  }
}

template <typename Dtype>
void FrcnnRoiDataLayer<Dtype>::FlipRois(vector<vector<float> > &rois, const float cols) {
  for (int i = 0; i < rois.size(); i++) {
    CHECK_GE(rois[i][DataPrepare::X1], 0 ) << "rois[i][DataPrepare::X1] : " << rois[i][DataPrepare::X1];
    CHECK_LT(rois[i][DataPrepare::X2], cols ) << "rois[i][DataPrepare::X2] : " << rois[i][DataPrepare::X2];
    float old_x1 = rois[i][DataPrepare::X1];
    float old_x2 = rois[i][DataPrepare::X2];
    rois[i][DataPrepare::X1] = cols - old_x2 - 1; 
    rois[i][DataPrepare::X2] = cols - old_x1 - 1; 
  }
}

// This function is called on prefetch thread
template <typename Dtype>
void FrcnnRoiDataLayer<Dtype>::load_batch(Batch<Dtype> *batch) {
  // At each iteration, Give Batch images and
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;

  const vector<float> scales = FrcnnParam::scales;
  const bool mirror = FrcnnParam::use_flipped;
  //const int batch_size = 1;
  // fyk modify for supporting batch > 1
  const int batch_size = FrcnnParam::IMS_PER_BATCH;
  bool do_mirror = mirror && PrefetchRand() % 2 && this->phase_ == TRAIN;

  //timer.Start();
  CHECK_EQ(roi_database_.size(), image_database_.size())
      << "image and roi size abnormal";

  // Select id for batch -> <0 if fliped
  ShuffleImages();//fyk: lines_id_ start from 0 + batchsize,the heading images is skiped in the first batch,but no need to worry about this
  CHECK(lines_id_ < lines_.size() && lines_id_ >= 0) << "select error line id : " << lines_id_;
  // fyk modify for supporting batch > 1,start batch loop
 vector<cv::Mat> ims, orig_ims;
 vector<float> im_scales;
 int max_rows=0,max_cols=0;
 for (int batch_i = 0; batch_i < batch_size; batch_i++) {
  int b_lines_id = lines_id_ + batch_i;
  if(b_lines_id >= lines_.size()) b_lines_id %= lines_.size();//we can also break here
  int index = lines_[b_lines_id];
  float max_short = scales[PrefetchRand() % scales.size()];

  //read_time += timer.MicroSeconds();

  // Prepare Image and labels;
  timer.Start();
  cv::Mat cv_img;
  if (this->cache_images_) {
    pair<std::string, Datum> image_cached = image_database_cache_[index];
    cv_img = DecodeDatumToCVMat(image_cached.second, true);
  } else {
    cv_img = cv::imread(image_database_[index], CV_LOAD_IMAGE_COLOR);
    if (!cv_img.data) {
      LOG(FATAL) << "Could not open or find file " << image_database_[index];
      return;
    }
  }
  //fyk : do equlize_hist,only for 3-channel
  int he_case = FrcnnParam::use_hist_equalize;
  switch(he_case) {
  case 1:
        cv_img = equalizeIntensityHist(cv_img);
        break;
  case 2:
        cv_img = equalizeChannelHist(cv_img);
        break;
  default:
        break;
  }
  //fyk end
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
  orig_ims.push_back(src.clone());
  float im_scale = Frcnn::get_scale_factor(src.cols, src.rows, max_short, max_long_);
  //fyk: check decimation or zoom,use different method
  //fyk: note that im_scale is a scale factor,and cv::Size()=0,so cv::resize will keep the original aspect ratio
  if (im_scale < 1)
  	cv::resize(src, src, cv::Size(), im_scale, im_scale, cv::INTER_AREA );
  else //fyk end
  cv::resize(src, src, cv::Size(), im_scale, im_scale);
  // same as py-faster-rcnn im_list_to_blob
  ims.push_back(src.clone());
  im_scales.push_back(im_scale);
  max_rows = max_rows > src.rows? max_rows : src.rows;
  max_cols = max_cols > src.cols? max_cols : src.cols;
 }//end loop for batch 
  // resize data
  //batch->data_.Reshape(batch_size, 3, src.rows, src.cols);
  batch->data_.Reshape(batch_size, 3, max_rows, max_cols);//include padding
  Dtype *top_data = batch->data_.mutable_cpu_data();
  caffe_set(batch->data_.count(), Dtype(0), top_data);//init padding to 0
 for (int batch_i = 0; batch_i < batch_size; batch_i++) {
  int img_offset = batch_i * 3 * max_rows * max_cols;
  cv::Mat src = ims[batch_i];
  for (int r = 0; r < src.rows; r++) {
    for (int c = 0; c < src.cols; c++) {
      int cv_offset = (r * src.cols + c) * 3;//src img data idx
      int blob_shift = r * max_cols + c;
      //fyk change from src.* to max_* for adding padding
      top_data[img_offset + 0 * max_rows * max_cols + blob_shift] = reinterpret_cast<float *>(src.data)[cv_offset + 0];
      top_data[img_offset + 1 * max_rows * max_cols + blob_shift] = reinterpret_cast<float *>(src.data)[cv_offset + 1];
      top_data[img_offset + 2 * max_rows * max_cols + blob_shift] = reinterpret_cast<float *>(src.data)[cv_offset + 2];
    }
  }
 }//end loop for batch

//  top_label[0] = src.rows; // height
//  top_label[1] = src.cols; // width
//  top_label[2] = im_scale; // im_scale: used to filter min size
//  top_label[3] = 0;
//  top_label[4] = 0;

  // fyk modify for supporting batch > 1
  vector<vector<float> > rois;
  vector<float> ims_info;
  vector<int> im_inds;
  int channels = 0;
  for (int i = 0; i < batch_size; i++) {
    cv::Mat scaled_img = ims[i];
    cv::Mat orig_img = orig_ims[i];
    int b_lines_id = lines_id_ + i;
    if(b_lines_id >= lines_.size()) b_lines_id %= lines_.size();
    int index = lines_[b_lines_id];
    vector<vector<float> > _rois = roi_database_[index];
    // Flip
    if (do_mirror) {
        //FlipRois(rois, cv_img.cols);
        FlipRois(_rois, orig_img.cols);
    }
    // Check and Reset rois
    CheckResetRois(_rois, image_database_[index], orig_img.cols, orig_img.rows, im_scales[i]);
    const int roi_num = _rois.size() ;//rois number of index-th image
    //LOG(INFO) << i << " th image has GT num: " << roi_num;
    // notice that we save the original im_info instead of image info after padding, since we only need the info to guarantee the box inside image border in anchor target layer and proposal layer,.etc and the im_scale only be used in removing small boxes, it doesn't matter too much.
    //static const float arr[] = {(float)scaled_img.rows,(float)scaled_img.cols,(float)im_scales[i],(float)roi_num,(float)0};// size=5 for alignment. add padding
    // BUG fix! DO NOT USE STATIC VALUE, for it only initialize only once
    // there we still use scaled_img size instead of padded size
    const float arr[] = {(float)scaled_img.rows,(float)scaled_img.cols,(float)im_scales[i],(float)roi_num,(float)0};// size=5 for alignment. add padding
    vector<float> image_label (arr, arr + sizeof(arr) / sizeof(arr[0]) );//Conventional STL,can also use vector = {} in C++11
    channels += roi_num + 1;//rois number of index-th image plus 1 for im_info;
    // append
    rois.insert(rois.end(), _rois.begin(), _rois.end());
    ims_info.insert(ims_info.end(), image_label.begin(), image_label.end());
    vector<int> im_ind_ (roi_num, i); 
    im_inds.insert(im_inds.end(), im_ind_.begin(), im_ind_.end());
  }
  // fyk modify for supporting batch > 1
  batch->label_.Reshape(channels, 5, 1, 1);
//LOG(INFO) << "batch->label_.shape_string : " << batch->label_.shape_string();
  Dtype *top_label = batch->label_.mutable_cpu_data();
  //CHECK_EQ(rois.size(), channels-batch_size);
  // fyk save im_info,will push to top blob at forward()
  for (int i = 0; i < batch_size; i++) {
    top_label[5 * i + 0] = ims_info[5 * i + 0];
    top_label[5 * i + 1] = ims_info[5 * i + 1];
    top_label[5 * i + 2] = ims_info[5 * i + 2];
    top_label[5 * i + 3] = ims_info[5 * i + 3];
    top_label[5 * i + 4] = ims_info[5 * i + 4];
  }

  for (int i = batch_size; i < channels; i++) {
    CHECK_EQ(rois[i-batch_size].size(), DataPrepare::NUM);
    //since i have scaled the rois,there is no need to scale.
    top_label[5 * i + 0] = rois[i-batch_size][DataPrepare::X1] * im_scales[i-batch_size]; // x1
    top_label[5 * i + 1] = rois[i-batch_size][DataPrepare::Y1] * im_scales[i-batch_size]; // y1
    top_label[5 * i + 2] = rois[i-batch_size][DataPrepare::X2] * im_scales[i-batch_size]; // x2
    top_label[5 * i + 3] = rois[i-batch_size][DataPrepare::Y2] * im_scales[i-batch_size]; // y2
    top_label[5 * i + 4] = rois[i-batch_size][DataPrepare::LABEL];         // label
    CHECK_LT(top_label[5 * i + 4], FrcnnParam::n_classes);//fyk
    //fyk
    int h = ims[im_inds[i - batch_size]].rows;
    int w = ims[im_inds[i - batch_size]].cols;

    if (top_label[5 * i + 3] >= h) {
      //DLOG(INFO) << mirror << " row : " << src.rows << ",  col : " << src.cols << ", im_scale : " << im_scale << " | " << rois[i-batch_size][DataPrepare::Y2] << " , " << top_label[5 * i + 3];
	  DLOG(INFO) << mirror << rois[i-batch_size][DataPrepare::Y2] << " , " << top_label[5 * i + 3];
	  top_label[5 * i + 3] = h - 1;
    }
    if (top_label[5 * i + 2] >= w) {
      //DLOG(INFO) << mirror << " row : " << src.rows << ",  col : " << src.cols << ", im_scale : " << im_scale << " | " << rois[i-batch_size][DataPrepare::X2] << " , " << top_label[5 * i + 2];
	  DLOG(INFO) << mirror << rois[i-batch_size][DataPrepare::X2] << " , " << top_label[5 * i + 2];
      top_label[5 * i + 2] = w - 1;
    }
    if (top_label[5 * i + 0] < 0) {
      top_label[5 * i + 0] = 0;
      //DLOG(INFO) << mirror << " row : " << src.rows << ",  col : " << src.cols << ", im_scale : " << im_scale << " | " << rois[i-batch_size][DataPrepare::X2] << " , " << top_label[5 * i + 2];
	  DLOG(INFO) << mirror << rois[i-batch_size][DataPrepare::X2] << " , " << top_label[5 * i + 2];
    }
    if (top_label[5 * i + 1] < 0) {
      top_label[5 * i + 1] = 0;
      //DLOG(INFO) << mirror << " row : " << src.rows << ",  col : " << src.cols << ", im_scale : " << im_scale << " | " << rois[i-batch_size][DataPrepare::Y2] << " , " << top_label[5 * i + 3];
	  DLOG(INFO) << mirror << rois[i-batch_size][DataPrepare::Y2] << " , " << top_label[5 * i + 3];
    }
  }

  trans_time += timer.MicroSeconds();

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
void FrcnnRoiDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = this->prefetch_full_.pop("Data layer prefetch queue empty");
  DLOG(INFO) << "====================data layer batch:" << batch->data_.num();//fyk
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(), top[0]->mutable_cpu_data());
  if (this->output_labels_) {
    //caffe_copy(3, batch->label_.cpu_data(), top[1]->mutable_cpu_data());
    // fyk modify for supporting batch > 1
    const int batch_size = FrcnnParam::IMS_PER_BATCH;
    caffe_copy(batch_size * 5, batch->label_.cpu_data(), top[1]->mutable_cpu_data());
    // Reshape to loaded labels.
    top[2]->Reshape(batch->label_.num()-batch_size, batch->label_.channels(), batch->label_.height(), batch->label_.width());
    // Copy the labels.
    caffe_copy(batch->label_.count() - batch_size * 5, batch->label_.cpu_data() + batch_size * 5, top[2]->mutable_cpu_data());
  }
  this->prefetch_free_.push(batch);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(FrcnnRoiDataLayer, Forward);
#endif

INSTANTIATE_CLASS(FrcnnRoiDataLayer);
REGISTER_LAYER_CLASS(FrcnnRoiData);

} // namespace Frcnn

} // namespace caffe
