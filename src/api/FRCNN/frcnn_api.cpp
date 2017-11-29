#include "api/FRCNN/frcnn_api.hpp"

namespace FRCNN_API{

void Detector::preprocess(const cv::Mat &img_in, const int blob_idx) {
  const vector<Blob<float> *> &input_blobs = net_->input_blobs();
  CHECK(img_in.isContinuous()) << "Warning : cv::Mat img_out is not Continuous !";
  DLOG(ERROR) << "img_in (CHW) : " << img_in.channels() << ", " << img_in.rows << ", " << img_in.cols; 
  input_blobs[blob_idx]->Reshape(1, img_in.channels(), img_in.rows, img_in.cols);
  float *blob_data = input_blobs[blob_idx]->mutable_cpu_data();
  const int cols = img_in.cols;
  const int rows = img_in.rows;
  for (int i = 0; i < cols * rows; i++) {
    blob_data[cols * rows * 0 + i] =
        reinterpret_cast<float*>(img_in.data)[i * 3 + 0] ;// mean_[0]; 
    blob_data[cols * rows * 1 + i] =
        reinterpret_cast<float*>(img_in.data)[i * 3 + 1] ;// mean_[1];
    blob_data[cols * rows * 2 + i] =
        reinterpret_cast<float*>(img_in.data)[i * 3 + 2] ;// mean_[2];
  }
}

void Detector::preprocess(const vector<float> &data, const int blob_idx) {
  const vector<Blob<float> *> &input_blobs = net_->input_blobs();
  input_blobs[blob_idx]->Reshape(1, data.size(), 1, 1);
  float *blob_data = input_blobs[blob_idx]->mutable_cpu_data();
  std::memcpy(blob_data, &data[0], sizeof(float) * data.size());
}

void Detector::Set_Model(std::string &proto_file, std::string &model_file) {
  this->roi_pool_layer = - 1;
  net_.reset(new Net<float>(proto_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(model_file);
  mean_[0] = FrcnnParam::pixel_means[0];
  mean_[1] = FrcnnParam::pixel_means[1];
  mean_[2] = FrcnnParam::pixel_means[2];
  const vector<std::string>& layer_names = this->net_->layer_names();
  const std::string roi_name = "roi_pool";
  for (size_t i = 0; i < layer_names.size(); i++) {
    if (roi_name.size() > layer_names[i].size()) continue;
    if (roi_name == layer_names[i].substr(0, roi_name.size())) {
      CHECK_EQ(this->roi_pool_layer, -1) << "Previous roi layer : " << this->roi_pool_layer << " : " << layer_names[this->roi_pool_layer];
      this->roi_pool_layer = i;
    }
  }
  CHECK(this->roi_pool_layer >= 0 && this->roi_pool_layer < layer_names.size());
  DLOG(INFO) << "SET MODEL DONE, ROI POOLING LAYER : " << layer_names[this->roi_pool_layer];
  caffe::Frcnn::FrcnnParam::print_param();
}

vector<boost::shared_ptr<Blob<float> > > Detector::predict(const vector<std::string> blob_names) {
  DLOG(ERROR) << "FORWARD BEGIN";
  float loss;
  net_->Forward(&loss);
  vector<boost::shared_ptr<Blob<float> > > output;
  for (int i = 0; i < blob_names.size(); ++i) {
    output.push_back(this->net_->blob_by_name(blob_names[i]));
  }
  DLOG(ERROR) << "FORWARD END, Loss : " << loss;
  return output;
}

void Detector::predict(const cv::Mat &img_in, std::vector<caffe::Frcnn::BBox<float> > &results) {
  CHECK(FrcnnParam::iter_test == -1 || FrcnnParam::iter_test > 1) << "FrcnnParam::iter_test == -1 || FrcnnParam::iter_test > 1";
  if (FrcnnParam::iter_test == -1) {
    predict_original(img_in, results);
  } else {
    predict_iterative(img_in, results);
  }
}

void Detector::predict_original(const cv::Mat &img_in, std::vector<caffe::Frcnn::BBox<float> > &results) {

  CHECK(FrcnnParam::test_scales.size() == 1) << "Only single-image batch implemented";

  float scale_factor = caffe::Frcnn::get_scale_factor(img_in.cols, img_in.rows, FrcnnParam::test_scales[0], FrcnnParam::test_max_size);

  cv::Mat img;
  const int height = img_in.rows;
  const int width = img_in.cols;
  DLOG(INFO) << "height: " << height << " width: " << width;
  img_in.convertTo(img, CV_32FC3);
  for (int r = 0; r < img.rows; r++) {
    for (int c = 0; c < img.cols; c++) {
      int offset = (r * img.cols + c) * 3;
      reinterpret_cast<float *>(img.data)[offset + 0] -= this->mean_[0]; // B
      reinterpret_cast<float *>(img.data)[offset + 1] -= this->mean_[1]; // G
      reinterpret_cast<float *>(img.data)[offset + 2] -= this->mean_[2]; // R
    }
  }
  cv::resize(img, img, cv::Size(), scale_factor, scale_factor);

  std::vector<float> im_info(3);
  im_info[0] = img.rows;
  im_info[1] = img.cols;
  im_info[2] = scale_factor;

  DLOG(ERROR) << "im_info : " << im_info[0] << ", " << im_info[1] << ", " << im_info[2];
  this->preprocess(img, 0);
  this->preprocess(im_info, 1);

  vector<std::string> blob_names(3);
  blob_names[0] = "rois";
  blob_names[1] = "cls_prob";
  blob_names[2] = "bbox_pred";

  vector<boost::shared_ptr<Blob<float> > > output = this->predict(blob_names);
  boost::shared_ptr<Blob<float> > rois(output[0]);
  boost::shared_ptr<Blob<float> > cls_prob(output[1]);
  boost::shared_ptr<Blob<float> > bbox_pred(output[2]);

  const int box_num = bbox_pred->num();
  const int cls_num = cls_prob->channels();
  CHECK_EQ(cls_num , caffe::Frcnn::FrcnnParam::n_classes);
  results.clear();

  for (int cls = 1; cls < cls_num; cls++) { 
    vector<BBox<float> > bbox;
    for (int i = 0; i < box_num; i++) { 
      float score = cls_prob->cpu_data()[i * cls_num + cls];

      Point4f<float> roi(rois->cpu_data()[(i * 5) + 1]/scale_factor,
                     rois->cpu_data()[(i * 5) + 2]/scale_factor,
                     rois->cpu_data()[(i * 5) + 3]/scale_factor,
                     rois->cpu_data()[(i * 5) + 4]/scale_factor);

      Point4f<float> delta(bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 0],
                     bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 1],
                     bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 2],
                     bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 3]);

      Point4f<float> box = caffe::Frcnn::bbox_transform_inv(roi, delta);
      box[0] = std::max(0.0f, box[0]);
      box[1] = std::max(0.0f, box[1]);
      box[2] = std::min(width-1.f, box[2]);
      box[3] = std::min(height-1.f, box[3]);

      // BBox tmp(box, score, cls);
      // LOG(ERROR) << "cls: " << tmp.id << " score: " << tmp.confidence;
      // LOG(ERROR) << "roi: " << roi.to_string();
      bbox.push_back(BBox<float>(box, score, cls));
    }
    sort(bbox.begin(), bbox.end());
    vector<bool> select(box_num, true);
    // Apply NMS
    for (int i = 0; i < box_num; i++)
      if (select[i]) {
        if (bbox[i].confidence < FrcnnParam::test_score_thresh) break;
        for (int j = i + 1; j < box_num; j++) {
          if (select[j]) {
            if (get_iou(bbox[i], bbox[j]) > FrcnnParam::test_nms) {
              select[j] = false;
            }
          }
        }
        results.push_back(bbox[i]);
      }
  }

}

void Detector::predict_iterative(const cv::Mat &img_in, std::vector<caffe::Frcnn::BBox<float> > &results) {

  CHECK(FrcnnParam::test_scales.size() == 1) << "Only single-image batch implemented";
  CHECK(FrcnnParam::iter_test >= 1) << "iter_test should greater and queal than 1";

  float scale_factor = caffe::Frcnn::get_scale_factor(img_in.cols, img_in.rows, FrcnnParam::test_scales[0], FrcnnParam::test_max_size);

  cv::Mat img;
  const int height = img_in.rows;
  const int width = img_in.cols;
  DLOG(INFO) << "height: " << height << " width: " << width;
  img_in.convertTo(img, CV_32FC3);
  for (int r = 0; r < img.rows; r++) {
    for (int c = 0; c < img.cols; c++) {
      int offset = (r * img.cols + c) * 3;
      reinterpret_cast<float *>(img.data)[offset + 0] -= this->mean_[0]; // B
      reinterpret_cast<float *>(img.data)[offset + 1] -= this->mean_[1]; // G
      reinterpret_cast<float *>(img.data)[offset + 2] -= this->mean_[2]; // R
    }
  }
  cv::resize(img, img, cv::Size(), scale_factor, scale_factor);

  std::vector<float> im_info(3);
  im_info[0] = img.rows;
  im_info[1] = img.cols;
  im_info[2] = scale_factor;

  DLOG(INFO) << "im_info : " << im_info[0] << ", " << im_info[1] << ", " << im_info[2];
  this->preprocess(img, 0);
  this->preprocess(im_info, 1);

  vector<std::string> blob_names(3);
  blob_names[0] = "rois";
  blob_names[1] = "cls_prob";
  blob_names[2] = "bbox_pred";
  
  vector<boost::shared_ptr<Blob<float> > > output = this->predict(blob_names);
  boost::shared_ptr<Blob<float> > rois(output[0]);
  boost::shared_ptr<Blob<float> > cls_prob(output[1]);
  boost::shared_ptr<Blob<float> > bbox_pred(output[2]);

  const int box_num = bbox_pred->num();
  const int cls_num = cls_prob->channels();
  CHECK_EQ(cls_num , caffe::Frcnn::FrcnnParam::n_classes);

  int iter_test = FrcnnParam::iter_test;
  while (--iter_test) {
    vector<BBox<float> > new_rois;
    for (int i = 0; i < box_num; i++) { 
      int cls_mx = 1;
      for (int cls = 1; cls < cls_num; cls++) { 
        float score    = cls_prob->cpu_data()[i * cls_num + cls];
        float mx_score = cls_prob->cpu_data()[i * cls_num + cls_mx];
        if (score >= mx_score) {
          cls_mx = cls;
        }
      }

      Point4f<float> roi(rois->cpu_data()[(i * 5) + 1],
                         rois->cpu_data()[(i * 5) + 2],
                         rois->cpu_data()[(i * 5) + 3],
                         rois->cpu_data()[(i * 5) + 4]);
#if 0
      new_rois.push_back( roi );
#endif

      Point4f<float> delta(bbox_pred->cpu_data()[(i * cls_num + cls_mx) * 4 + 0],
                           bbox_pred->cpu_data()[(i * cls_num + cls_mx) * 4 + 1],
                           bbox_pred->cpu_data()[(i * cls_num + cls_mx) * 4 + 2],
                           bbox_pred->cpu_data()[(i * cls_num + cls_mx) * 4 + 3]);

      Point4f<float> box = caffe::Frcnn::bbox_transform_inv(roi, delta);
      box[0] = std::max(0.0f, box[0]);
      box[1] = std::max(0.0f, box[1]);
      box[2] = std::min(im_info[1]-1.f, box[2]);
      box[3] = std::min(im_info[0]-1.f, box[3]);

      new_rois.push_back(box);
    }
    rois->Reshape(new_rois.size(), 5, 1, 1);
    for (size_t index = 0; index < new_rois.size(); index++) {
      rois->mutable_cpu_data()[ index * 5 ] = 0;
      for (int j = 1; j < 5; j++) {
        rois->mutable_cpu_data()[ index * 5 + j ] = new_rois[index][j-1];
      }
    }
    this->net_->ForwardFrom( this->roi_pool_layer );
    DLOG(INFO) << "iter_test[" << iter_test << "] >>> rois shape : " << rois->shape_string() << "  |  cls_prob shape : " << cls_prob->shape_string() << " | bbox_pred : " << bbox_pred->shape_string();
  }
  
  results.clear();

  for (int cls = 1; cls < cls_num; cls++) { 
    vector<BBox<float> > bbox;
    for (int i = 0; i < box_num; i++) { 
      float score = cls_prob->cpu_data()[i * cls_num + cls];

      Point4f<float> roi(rois->cpu_data()[(i * 5) + 1]/scale_factor,
                     rois->cpu_data()[(i * 5) + 2]/scale_factor,
                     rois->cpu_data()[(i * 5) + 3]/scale_factor,
                     rois->cpu_data()[(i * 5) + 4]/scale_factor);

      Point4f<float> delta(bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 0],
                     bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 1],
                     bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 2],
                     bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 3]);

      Point4f<float> box = caffe::Frcnn::bbox_transform_inv(roi, delta);
      box[0] = std::max(0.0f, box[0]);
      box[1] = std::max(0.0f, box[1]);
      box[2] = std::min(width-1.f, box[2]);
      box[3] = std::min(height-1.f, box[3]);

      bbox.push_back(BBox<float>(box, score, cls));
    }
    sort(bbox.begin(), bbox.end());
    vector<bool> select(box_num, true);
    // Apply NMS
    for (int i = 0; i < box_num; i++)
      if (select[i]) {
        if (bbox[i].confidence < FrcnnParam::test_score_thresh) break;
        for (int j = i + 1; j < box_num; j++) {
          if (select[j]) {
            if (get_iou(bbox[i], bbox[j]) > FrcnnParam::test_nms) {
              select[j] = false;
            }
          }
        }
        results.push_back(bbox[i]);
      }
  }

}

} // FRCNN_API
