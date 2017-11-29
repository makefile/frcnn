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
#include "api/FRCNN/frcnn_api.hpp"
#include "api/FRCNN/rpn_api.hpp"

namespace API{

using std::vector;
using caffe::Blob;
using caffe::Net;
using caffe::Frcnn::FrcnnParam;
using caffe::Frcnn::Point4f;
using caffe::Frcnn::BBox;
using caffe::Frcnn::DataPrepare;
using FRCNN_API::Detector;
using FRCNN_API::Rpn_Det;

inline void Set_Config(std::string default_config) {
  caffe::Frcnn::FrcnnParam::load_param(default_config);
  caffe::Frcnn::FrcnnParam::print_param();
}

}
