// ------------------------------------------------------------------
// Xuanyi . Refer to Dong Jian
// 2016/04/01
// ------------------------------------------------------------------
#ifndef CAFFE_FRCNN_VIS_HPP_
#define CAFFE_FRCNN_VIS_HPP_

#include "caffe/FRCNN/util/frcnn_utils.hpp"

namespace caffe {

namespace Frcnn {

inline std::map<int,string> LoadVocClass(){
  std::string CLASSES[] = {"__background__",
           "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"};
  std::map<int,string> answer;
  for (int index = 0 ; index < 21; index++) {
    answer[index] = CLASSES[index]; 
  }
  return answer;
}

inline std::string GetClassName(const std::map<int,std::string> CLASS, int label) {
  if( CLASS.find(label) == CLASS.end() ){
    std::ostringstream text;
    text << "Unknow_Class_" << label;
    return text.str();
  } else {
    return CLASS.find(label)->second;
  }
}

inline std::map<int,string> LoadRpnClass() {
  std::map<int,string> answer;
  answer[0] = "background";
  answer[1] = "proposal";
  return answer;
}

template <typename Dtype>
void vis_detections(cv::Mat & frame, const std::vector<BBox<Dtype> >& ans, const std::map<int,std::string> CLASS); 

template <typename Dtype>
void vis_detections(cv::Mat & frame, const BBox<Dtype> ans, const std::map<int,std::string> CLASS); 

}  // namespace frcnn

}  // namespace caffe

#endif
