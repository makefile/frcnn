/**
    pybind api for faster rcnn detector
*/
#include <glog/logging.h>
//#include "boost/algorithm/string.hpp"
#include "caffe/util/benchmark.hpp"
#include "api/api.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for conversion of std::vector std::list etc, map to python list
#include <pybind11/numpy.h>
namespace py = pybind11;

class FRCNNDetector {
  public:
    FRCNNDetector(std::string &proto_file, std::string &weight_file, std::string &config_file, int gpu_id=0);
    // [[cls_id,x1,y1,x2,y2,confidence],]
    std::vector<std::vector<float> > predict(std::string &img_path);
    std::vector<std::vector<float> > predict_numpy(py::array_t<float> img_numpy);
    virtual void destroy(){delete _detector;} // release resources
  private:
    API::Detector *_detector;
};

// python bindings, see http://pybind11.readthedocs.io/en/master/classes.html
PYBIND11_MODULE(frcnn, m) {
    py::class_<FRCNNDetector>(m, "FRCNNDetector")
        .def(py::init<std::string &, std::string &, std::string &, int>()) //constructor
        .def("predict", &FRCNNDetector::predict)
        .def("predict_numpy", &FRCNNDetector::predict_numpy)
        .def("destroy", &FRCNNDetector::destroy);
}

/**
    gpu_id -1 for use cpu
*/
FRCNNDetector::FRCNNDetector(std::string &proto_file, std::string &weight_file, std::string &config_file, int gpu_id) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Run tool or show usage.int argc, char **argv, 
  //caffe::GlobalInit(&argc, &argv);
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

  API::Set_Config(config_file);
  _detector = new API::Detector (proto_file, weight_file);
}
std::vector<std::vector<float> > FRCNNDetector::predict(std::string &img_path) {
    std::vector<caffe::Frcnn::BBox<float> > results;
    caffe::Timer time_;
  
    cv::Mat image = cv::imread(img_path);
    time_.Start();
    _detector->predict(image, results);
    LOG(INFO) << "Predict " << img_path << " : " << results.size() << " objects, cost " << time_.MilliSeconds() << " ms.";
    time_.Stop();
    
    std::vector<std::vector<float> > ret;
    for (size_t obj = 0; obj < results.size(); obj++) {
      LOG(INFO) << results[obj].to_string();
      std::vector<float> t(6); // cls_id,x1,y1,x2,y2,confidence
      t[0] = results[obj].id;
      for(int j=0;j<4;j++) t[j+1] = results[obj][j];
      t[5] = results[obj].confidence;
      ret.push_back(t);
    }
    return ret;
}
std::vector<std::vector<float> > FRCNNDetector::predict_numpy(py::array_t<float> img_numpy) {
    std::vector<caffe::Frcnn::BBox<float> > results;
    caffe::Timer time_;
    auto buf = img_numpy.request();
    float *ptr = (float *) buf.ptr;
    // img_numpy is of shape (h,w,c)
    int rows = buf.shape[0];
    int cols = buf.shape[1];
    // C++: Mat::Mat(int rows, int cols, int type, void* data, size_t step=AUTO_STEP)
    // this constructor only creates headers and does not copy data, and OpenCV data is row major
    cv::Mat image(rows, cols, CV_32FC3, ptr);
    time_.Start();
    _detector->predict(image, results);
    time_.Stop();

    std::vector<std::vector<float> > ret;
    for (size_t obj = 0; obj < results.size(); obj++) {
      //LOG(INFO) << results[obj].to_string();
      std::vector<float> t(6); // cls_id,x1,y1,x2,y2,confidence
      t[0] = results[obj].id;
      for(int j=0;j<4;j++) t[j+1] = results[obj][j];
      t[5] = results[obj].confidence;
      ret.push_back(t);
    }
    return ret;
}
