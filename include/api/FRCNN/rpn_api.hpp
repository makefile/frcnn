#include <vector>
#include <string>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"

namespace FRCNN_API{

using std::vector;
using caffe::Blob;
using caffe::Net;
using caffe::Frcnn::FrcnnParam;
using caffe::Frcnn::Point4f;
using caffe::Frcnn::BBox;

class Rpn_Det {
public:
  Rpn_Det(std::string &proto_file, std::string &model_file){
    Set_Model(proto_file, model_file);
  }
  void Set_Model(std::string &proto_file, std::string &model_file);
  void predict(const cv::Mat &img_in, vector<BBox<float> > &results);
private:
  void preprocess(const cv::Mat &img_in, const int blob_idx);
  void preprocess(const vector<float> &data, const int blob_idx);
  vector<boost::shared_ptr<Blob<float> > > predict(const vector<std::string> blob_names);
  boost::shared_ptr<Net<float> > net_;
  float mean_[3];
};

}
