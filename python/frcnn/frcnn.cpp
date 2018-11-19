/**
    pybind api for faster rcnn & YOLOv3 detector
*/
#include <glog/logging.h>
//#include "boost/algorithm/string.hpp"
#include "caffe/util/benchmark.hpp"
#include "api/api.hpp"
//for yolo v3
#include "caffe/YOLO/yolo_layer.h"
#include "caffe/YOLO/image.h"
image cvmat_to_image(cv::Mat &mat);

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for conversion of std::vector std::list etc, map to python list
#include <pybind11/numpy.h>
namespace py = pybind11;

class FRCNNDetector {
  public:
    FRCNNDetector(std::string &proto_file, std::string &weight_file, std::string &config_file, int gpu_id=0);
    // [[cls_id,x1,y1,x2,y2,confidence],]
    std::vector<std::vector<float> > predict(std::string &img_path, int gpu_id);
    std::vector<std::vector<float> > predict_numpy(py::array_t<float> img_numpy, int gpu_id);
    std::vector<std::vector<float> > predict_yolov3_numpy(py::array_t<float> img_numpy, int gpu_id);
    virtual void destroy(){delete _detector;} // release resources
  private:
    API::Detector *_detector;
    int gpu_id = 0;
    void set_mode(int gpu_id);
};

class YOLOv3Detector {
  public:
    YOLOv3Detector(std::string &proto_file, std::string &weight_file, int gpu_id=0);
    // [[cls_id,x1,y1,x2,y2,confidence],]
    //std::vector<std::vector<float> > predict(std::string &img_path, int gpu_id);
    std::vector<std::vector<float> > predict_numpy(py::array_t<float> img_numpy, int gpu_id, int classes);
  private:
    shared_ptr<Net<float> > net;
    int gpu_id = 0;
    void set_mode(int gpu_id);
};

// python bindings, see http://pybind11.readthedocs.io/en/master/classes.html
PYBIND11_MODULE(frcnn, m) {
    py::class_<FRCNNDetector>(m, "FRCNNDetector")
        .def(py::init<std::string &, std::string &, std::string &, int>()) //constructor
        .def("predict", &FRCNNDetector::predict)
        .def("predict_numpy", &FRCNNDetector::predict_numpy)
        .def("destroy", &FRCNNDetector::destroy);

    py::class_<YOLOv3Detector>(m, "YOLOv3Detector")
        .def(py::init<std::string &, std::string &, int>()) //constructor
        //.def("predict", &FRCNNDetector::predict)
        .def("predict_numpy", &YOLOv3Detector::predict_numpy);
}

/**
    gpu_id -1 for use cpu
*/
FRCNNDetector::FRCNNDetector(std::string &proto_file, std::string &weight_file, std::string &config_file, int gpu_id) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Run tool or show usage.int argc, char **argv, 
  //caffe::GlobalInit(&argc, &argv);
  set_mode(gpu_id);

  API::Set_Config(config_file);
  _detector = new API::Detector (proto_file, weight_file);
}
std::vector<std::vector<float> > FRCNNDetector::predict(std::string &img_path, int gpu_id) {
    set_mode(gpu_id);
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
std::vector<std::vector<float> > FRCNNDetector::predict_numpy(py::array_t<float> img_numpy, int gpu_id) {
    set_mode(gpu_id);
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
    LOG(INFO) << "Predict " << results.size() << " objects, cost " << time_.MilliSeconds() << " ms.";
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
void FRCNNDetector::set_mode(int gpu_id) {
  if (gpu_id >= 0) {
#ifndef CPU_ONLY
    caffe::Caffe::SetDevice(gpu_id);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif
  } else {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }
}


/**
    gpu_id -1 for use cpu
*/
YOLOv3Detector::YOLOv3Detector(std::string &proto_file, std::string &weight_file, int gpu_id) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  set_mode(gpu_id);
  /* Load the network. */
  net.reset(new Net<float>(proto_file, caffe::TEST));
  net->CopyTrainedLayersFrom(weight_file);
}
std::vector<std::vector<float> > YOLOv3Detector::predict_numpy(py::array_t<float> img_numpy, int gpu_id, int classes) {
    set_mode(gpu_id);
    Blob<float> *input_data_blobs = net->input_blobs()[0];
    std::vector<caffe::Frcnn::BBox<float> > results;
    caffe::Timer time_;
    auto buf = img_numpy.request();
    float *ptr = (float *) buf.ptr;
    // img_numpy is of shape (h,w,c)
    int rows = buf.shape[0];
    int cols = buf.shape[1];
    // C++: Mat::Mat(int rows, int cols, int type, void* data, size_t step=AUTO_STEP)
    // this constructor only creates headers and does not copy data, and OpenCV data is row major
    cv::Mat img_in(rows, cols, CV_32FC3, ptr);
    image im = cvmat_to_image(img_in);
    //cv::resize(img_in, img_in, cv::Size(input_data_blobs->width(), input_data_blobs->height()));
    // resize image with unchanged aspect ratio using padding
    image sized = letterbox_image(im,input_data_blobs->width(),input_data_blobs->height());
    float *blob_data = input_data_blobs->mutable_cpu_data();
    int resize_area = input_data_blobs->width() * input_data_blobs->height();
    int input_size = input_data_blobs->channels() * resize_area;
    std::memcpy(blob_data, sized.data, sizeof(float) * input_size);
    /* LOG(INFO) << "predict yolo " << rows << " " << cols << " " << resize_area;
    for (int i = 0; i < resize_area; i++) {
      // notice that yolo need input pixel range 0~1.
      blob_data[resize_area * 0 + i] =
          reinterpret_cast<float*>(img_in.data)[i * 3 + 0] / 255.;
      blob_data[resize_area * 1 + i] =
          reinterpret_cast<float*>(img_in.data)[i * 3 + 1] / 255.;
      blob_data[resize_area * 2 + i] =
          reinterpret_cast<float*>(img_in.data)[i * 3 + 2] / 255.;
    } */
    time_.Start();
    //_detector->predict(image, results);
    float loss;
    net->Forward(&loss); // thus can forward any times
    time_.Stop();
    vector<Blob<float>*> blobs;
    Blob<float>* out_blob1 = net->output_blobs()[1];
    blobs.push_back(out_blob1);
    Blob<float>* out_blob2 = net->output_blobs()[2];
    blobs.push_back(out_blob2);
    Blob<float>* out_blob3 = net->output_blobs()[0];
    blobs.push_back(out_blob3);
    
    //int classes = 80;
    float thresh = 0.5;
    float nms = 0.3;
    int nboxes = 0;
    detection *dets = get_detections(blobs,cols,rows,input_data_blobs->width(), input_data_blobs->height(), &nboxes, classes, thresh, nms);
    //LOG(INFO) << "Predict " << nboxes << " objects, cost " << time_.MilliSeconds() << " ms.";

    std::vector<std::vector<float> > ret;
    int i,j;
    for(i=0; i < nboxes;++i){
        int cls = -1;
        for(j=0;j<classes;++j){
            if(dets[i].prob[j] > thresh){
                if(cls < 0){
                    cls = j;
                    break;
                }
            }
        }
        if(cls >= 0){
            box b = dets[i].bbox;
            int left  = (b.x-b.w/2.)*cols;
            int right = (b.x+b.w/2.)*cols;
            int top   = (b.y-b.h/2.)*rows;
            int bot   = (b.y+b.h/2.)*rows;
            std::vector<float> t(6); // cls_id,x1,y1,x2,y2,confidence
            t[0] = cls + 1; // yolo class id start from 0, however our system start from 1, usually 0 stands for background class
            t[1] = left; t[2] = top; t[3] = right; t[4] = bot; t[5] = dets[i].prob[j];
            ret.push_back(t);
        }
    }
    // free memory
    free_detections(dets,nboxes);
    free_image(im);
    free_image(sized);

    return ret;
}
void YOLOv3Detector::set_mode(int gpu_id) {
  if (gpu_id >= 0) {
#ifndef CPU_ONLY
    caffe::Caffe::SetDevice(gpu_id);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif
  } else {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }
}

