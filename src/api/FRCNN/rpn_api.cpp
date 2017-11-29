#include "api/FRCNN/rpn_api.hpp"

namespace FRCNN_API{

void Rpn_Det::preprocess(const cv::Mat &img_in, const int blob_idx) {
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

void Rpn_Det::preprocess(const vector<float> &data, const int blob_idx){
  const vector<Blob<float> *> &input_blobs = net_->input_blobs();
  input_blobs[blob_idx]->Reshape(1, data.size(), 1, 1);
  float *blob_data = input_blobs[blob_idx]->mutable_cpu_data();
  std::memcpy(blob_data, &data[0], sizeof(float) * data.size());
}

void Rpn_Det::Set_Model(std::string &proto_file, std::string &model_file){
  net_.reset(new Net<float>(proto_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(model_file);
  mean_[0] = FrcnnParam::pixel_means[0];
  mean_[1] = FrcnnParam::pixel_means[1];
  mean_[2] = FrcnnParam::pixel_means[2];
  LOG(INFO) << "SET MODEL DONE";
}

vector<boost::shared_ptr<Blob<float> > > Rpn_Det::predict(const vector<std::string> blob_names) {
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

void Rpn_Det::predict(const cv::Mat &img_in, std::vector<caffe::Frcnn::BBox<float> > &results){

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

  vector<std::string> blob_names(2);
  blob_names[0] = "rois";
  blob_names[1] = "scores";

  vector<boost::shared_ptr<Blob<float> > > output = this->predict(blob_names);
  boost::shared_ptr<Blob<float> > rois(output[0]);
  boost::shared_ptr<Blob<float> > scores(output[1]);

  DLOG(INFO) << "rois  : " << rois->num() << ", " << rois->channels() << ", " << rois->height() << ", " << rois->width();
  DLOG(INFO) << "score : " << scores->num() << ", " << scores->channels() << ", " << scores->height() << ", " << scores->width();

  const int box_num = rois->num();
  results.clear();

  for (int i = 0; i < box_num; i++) { 
    float score = scores->data_at(i, 0, 0, 0);

    if (score < caffe::Frcnn::FrcnnParam::test_score_thresh) break;
 
    Point4f<float> roi(rois->cpu_data()[(i * 5) + 1]/scale_factor,
            rois->cpu_data()[(i * 5) + 2]/scale_factor,
            rois->cpu_data()[(i * 5) + 3]/scale_factor,
            rois->cpu_data()[(i * 5) + 4]/scale_factor);

    roi[0] = std::max(0.0f, roi[0]);
    roi[1] = std::max(0.0f, roi[1]);
    roi[2] = std::min(width-1.f, roi[2]);
    roi[3] = std::min(height-1.f, roi[3]);

    results.push_back(BBox<float>(roi, score, 1));
  }

}

} // FRCNN_API
