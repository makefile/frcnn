#include <fstream>
#include <sstream>
#include "caffe/FRCNN/util/frcnn_vis.hpp"

namespace caffe {

namespace Frcnn {

template <typename Dtype>
void vis_detections(cv::Mat & frame, const std::vector<BBox<Dtype> >& ans, const std::map<int,std::string> CLASS) { 
  for(size_t i = 0 ; i < ans.size() ; i++) {
    cv::rectangle(frame, cv::Point(ans[i][0],ans[i][1]) , cv::Point(ans[i][2],ans[i][3]) , cv::Scalar(255,255,255) );
    std::ostringstream text;
    text << GetClassName(CLASS, ans[i].id) << "  :  " << ans[i].confidence;
    cv::putText(frame, text.str() , cv::Point(ans[i][0],ans[i][1]-18) , 0 , 0.6 , cv::Scalar(0,255,0) );
  }   
}

template void vis_detections(cv::Mat & frame, const std::vector<BBox<float> >& ans, const std::map<int,std::string> CLASS);
template void vis_detections(cv::Mat & frame, const std::vector<BBox<double> >& ans, const std::map<int,std::string> CLASS);

template <typename Dtype>
void vis_detections(cv::Mat & frame, const BBox<Dtype> ans, const std::map<int,std::string> CLASS) { 
  std::vector<BBox<Dtype> > vec_ans;
  vec_ans.push_back( ans );
  vis_detections(frame, vec_ans, CLASS);
}

template void vis_detections(cv::Mat & frame, const BBox<float> ans, const std::map<int,std::string> CLASS);
template void vis_detections(cv::Mat & frame, const BBox<double> ans, const std::map<int,std::string> CLASS);

} // Frcnn

} // caffe 
