#include <gflags/gflags.h>
#include <glog/logging.h>
#include "boost/algorithm/string.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/signal_handler.h"
#include "api/api.hpp"

DEFINE_string(gpu, "", 
    "Optional; run in GPU mode on the given device ID, Empty is CPU");
DEFINE_string(model, "", 
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "", 
    "Trained Model By Faster RCNN End-to-End Pipeline.");
DEFINE_string(default_c, "", 
    "Default config file path.");
DEFINE_string(image_list, "", 
    "Optional;Test images list."); 
DEFINE_string(image_root, "", 
    "Optional;Test images root directory."); 
DEFINE_string(out_file, "", 
    "Optional;Output images file."); 

using std::string;
using std::vector;

inline string INT(float x) { char A[100]; sprintf(A,"%.1f",x); return string(A);};
inline string FloatToString(float x) { char A[100]; sprintf(A,"%.4f",x); return string(A);};
float mean_[3];

void Set_Model(boost::shared_ptr<caffe::Net<float> >& net_, std::string &proto_file, std::string &model_file) {
  net_.reset(new caffe::Net<float>(proto_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(model_file);
  ::mean_[0] = caffe::Frcnn::FrcnnParam::pixel_means[0];
  ::mean_[1] = caffe::Frcnn::FrcnnParam::pixel_means[1];
  ::mean_[2] = caffe::Frcnn::FrcnnParam::pixel_means[2]; 
  DLOG(INFO) << "SET MODEL DONE";
  caffe::Frcnn::FrcnnParam::print_param();
} 

vector<float> predict(boost::shared_ptr<caffe::Net<float> >& net_, const cv::Mat &img_in, const vector<caffe::Frcnn::BBox<float> > results) {
  CHECK(caffe::Frcnn::FrcnnParam::test_scales.size() == 1) << "Only single-image batch implemented";
  float scale_factor = caffe::Frcnn::get_scale_factor(img_in.cols, img_in.rows, caffe::Frcnn::FrcnnParam::test_scales[0], caffe::Frcnn::FrcnnParam::test_max_size);
  cv::Mat img;
  const int height = img_in.rows;
  const int width = img_in.cols;
  DLOG(INFO) << "height: " << height << " width: " << width;
  img_in.convertTo(img, CV_32FC3);
  for (int r = 0; r < img.rows; r++) {
    for (int c = 0; c < img.cols; c++) {
      int offset = (r * img.cols + c) * 3;
      reinterpret_cast<float *>(img.data)[offset + 0] -= ::mean_[0]; // B
      reinterpret_cast<float *>(img.data)[offset + 1] -= ::mean_[1]; // G
      reinterpret_cast<float *>(img.data)[offset + 2] -= ::mean_[2]; // R
    }
  }
  cv::resize(img, img, cv::Size(), scale_factor, scale_factor);
  float im_info[3];
  im_info[0] = img.rows;
  im_info[1] = img.cols;
  im_info[2] = scale_factor;

  CHECK(img.isContinuous()) << "Warning : cv::Mat img is not Continuous !";
  DLOG(ERROR) << "img (CHW) : " << img.channels() << ", " << img.rows << ", " << img.cols;
  boost::shared_ptr<caffe::Blob<float> > image_blob = net_->blob_by_name("data");
  image_blob->Reshape(1, img.channels(), img.rows, img.cols);
  const int cols = img.cols;
  const int rows = img.rows;
  for (int i = 0; i < cols * rows; i++) {
    image_blob->mutable_cpu_data()[cols * rows * 0 + i] =
        reinterpret_cast<float*>(img.data)[i * 3 + 0] ;// mean_[0]; 
    image_blob->mutable_cpu_data()[cols * rows * 1 + i] =
        reinterpret_cast<float*>(img.data)[i * 3 + 1] ;// mean_[1];
    image_blob->mutable_cpu_data()[cols * rows * 2 + i] =
        reinterpret_cast<float*>(img.data)[i * 3 + 2] ;// mean_[2];
  }

  boost::shared_ptr<caffe::Blob<float> > info_blob = net_->blob_by_name("im_info");
  info_blob->Reshape(1, 3, 1, 1);
  std::memcpy(info_blob->mutable_cpu_data(), im_info, sizeof(float) * 3);
  boost::shared_ptr<caffe::Blob<float> > rois_blob = net_->blob_by_name("rois");
  rois_blob->Reshape(results.size(), 5, 1, 1);
  for (size_t index = 0; index < results.size(); index++) {
    rois_blob->mutable_cpu_data()[ index *5 + 0 ] = 0;
    rois_blob->mutable_cpu_data()[ index *5 + 1 ] = std::max(0.f, results[index][0] * scale_factor);
    rois_blob->mutable_cpu_data()[ index *5 + 2 ] = std::max(0.f, results[index][1] * scale_factor);
    rois_blob->mutable_cpu_data()[ index *5 + 3 ] = std::min(im_info[1]-1.f, results[index][2] * scale_factor);
    rois_blob->mutable_cpu_data()[ index *5 + 4 ] = std::min(im_info[0]-1.f, results[index][3] * scale_factor);
  }
  float loss;
  net_->Forward(&loss);
  boost::shared_ptr<caffe::Blob<float> > cls_prob = net_->blob_by_name("cls_prob");
  const int cls_num = cls_prob->channels();
  vector<float> answer(results.size());
  CHECK_EQ(int(results.size()), cls_prob->num());
  for (size_t index = 0; index < results.size(); index++) {
    const int cls = results[index].id;
    CHECK_GT(cls, 0); CHECK_LT(cls, cls_num);
    answer[index] = cls_prob->cpu_data()[index * cls_num + cls];
  }
  DLOG(INFO) << "FORWARD END, LOSS : " << loss;
  return answer;
}

int main(int argc, char** argv){
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: demo_frcnn_api <args>\n\n"
      "args:\n"
      "  --gpu          7       use 7-th gpu device, default is cpu model\n"
      "  --model        file    protocol buffer text file\n"
      "  --weights      file    Trained Model\n"
      "  --default_c    file    Default Config File\n"
      "  --image_list   file    input image list\n"
      "  --image_root   file    input image dir\n"
      "  --out_file     file    output amswer file");

  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  CHECK( FLAGS_gpu.size() == 0 || FLAGS_gpu.size() == 1 || (FLAGS_gpu.size()==2&&FLAGS_gpu=="-1")) << "Can only support one gpu or none or -1(for cpu)";
  int gpu_id = -1;
  if( FLAGS_gpu.size() > 0 )
    gpu_id = boost::lexical_cast<int>(FLAGS_gpu);

  if (gpu_id >= 0) {
#ifndef CPU_ONLY
    caffe::Caffe::SetDevice(gpu_id);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
#else
    LOG(FATAL) << "CPU ONLY MODEL, BUT PROVIDE GPU ID";
#endif
  } else {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }

  boost::shared_ptr<caffe::Net<float> > net_;
  string proto_file             = FLAGS_model.c_str();
  string model_file             = FLAGS_weights.c_str();
  string default_config_file    = FLAGS_default_c.c_str();

  const string image_list = FLAGS_image_list.c_str();
  const string image_root = FLAGS_image_root.c_str();
  const string out_file = FLAGS_out_file.c_str();

  API::Set_Config(default_config_file);
  Set_Model(net_, proto_file, model_file);

  LOG(INFO) << "image list     : " << image_list;
  LOG(INFO) << "output file    : " << out_file;
  LOG(INFO) << "image_root     : " << image_root;
  std::ifstream infile(image_list.c_str());
  std::ofstream otfile(out_file.c_str());
  int count = 0;
  string shot_dir;
  int frames;
  string HASH, image;
  int ids_;
  while ( infile >> HASH >> ids_ >> shot_dir >> frames ) {
    CHECK(HASH == "#");
    CHECK(ids_ >= 0);
    otfile << "#\t" << ids_ << "\t" << shot_dir << "\t" << frames << std::endl;
    CHECK_GE(frames, 0);
    for (int ii = 0; ii < frames; ii++) {
      int boxes_num = 0;
      infile >> HASH >> image >> boxes_num;
      otfile << "&\t" << image << "\t" << boxes_num << std::endl;
      CHECK(HASH == "&");
      CHECK(image.find(".jpeg") != string::npos);
      if (boxes_num == 0) {
        LOG(INFO) << "Handle " << count << " th shot[" << shot_dir << "] : " << ii << " / " << frames << " frame : " << image << " -> " << boxes_num << " boxes";
        continue;
      }
      cv::Mat cv_image = cv::imread(image_root + "/" + shot_dir + "/" + image);
      vector<caffe::Frcnn::BBox<float> > results;
      for (int ibs = 0; ibs < boxes_num; ibs++) {
        int id;
        float x1, y1, x2, y2, conf;
        infile >> id >> x1 >> y1 >> x2 >> y2 >> conf;
        results.push_back(caffe::Frcnn::BBox<float>(x1, y1, x2, y2, 1, id));
      }
      vector<float> scores = predict(net_, cv_image, results);
      CHECK_EQ(scores.size(), results.size());
      for (size_t obj = 0; obj < scores.size(); obj++) {
        otfile << FloatToString(scores[obj]) << std::endl;
      }
      LOG(INFO) << "Handle " << count << " th shot[" << shot_dir << "] : " << ii << " / " << frames << " frame : " << image << " -> " << results.size() << " boxes";
    }
    count ++;
  }
  infile.close();
  otfile.close();
  return 0;
}
