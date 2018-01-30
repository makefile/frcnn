#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "rbbox_overlaps.hpp"

namespace caffe {

namespace Frcnn {

INSTANTIATE_CLASS(Point4f);
INSTANTIATE_CLASS(Point5f);//fyk 
INSTANTIATE_CLASS(BBox);
INSTANTIATE_CLASS(RBBox);

#define USE_GPU_NMS //fyk: accelerate
//get rotatedRectangle IoU by opencv
template <typename Dtype>
Dtype skew_iou(const Point5f<Dtype> &A, const Point5f<Dtype> &B) {
  //I met a bug in OpenCV after train 5w iters for two RotatedRect are very similar
  //see http://answers.opencv.org/question/74697/cvrotatedrectangleintersection-assertion-failed/.
  //the trick is to round the values to 1 decimal place or maybe 0 decimal place.
  if((fabs(A[0] - B[0]) < 1e-5) && (fabs(A[1] - B[1]) < 1e-5) && (fabs(A[2] - B[2]) < 1e-5) && (fabs(A[3] - B[3]) < 1e-5) && (fabs(A[4] - B[4]) < 1e-5)) {
      return 1.0;
  }
  // cx,cy,w,h,theta(anti-clockwise,rad)
  float region1[] = {(float)A[0],(float)A[1],(float)A[2],(float)A[3],(float)A[4]};
  float region2[] = {(float)B[0],(float)B[1],(float)B[2],(float)B[3],(float)B[4]};
  float result = rotateRectIoU(region1, region2);
  /*The following code uses OpenCV3 rotatedRectangleIntersection, which is defined in OpenCV source code of modules/imgproc/src/intersection.cpp
  Dtype ax = A[0];
  Dtype ay = A[1];
  Dtype aw = A[2];
  Dtype ah = A[3];
  Dtype bx = B[0];
  Dtype by = B[1];
  Dtype bw = B[2];
  Dtype bh = B[3];
  Dtype areaA = aw * ah;
  Dtype areaB = bw * bh;
  //opencv use clock-wise angle, but our annotations use reverse. and remember to use angle instead of rad.
  cv::RotatedRect rA(cv::Point2f(ax,ay),cv::Size2f(aw,ah), - A[4] * 180.0/M_PI);
  cv::RotatedRect rB(cv::Point2f(bx,by),cv::Size2f(bw,bh), - B[4] * 180.0/M_PI);
  std::vector<cv::Point2f> vertices;
  Dtype result = 0.0;
  try {
     cv::rotatedRectangleIntersection(rA, rB, vertices);
     Dtype inter = 0;
     if(vertices.size() != 0){
       std::vector<cv::Point2f> hull;
       cv::convexHull(vertices,hull);
       inter = contourArea(hull);
     }
     result = inter / (areaA + areaB - inter);
     if(result < 0) {
         result = 0.0;
     }
  } catch(std::exception& e) {
    //when rotatedRectangleIntersection get vertices greater than 8,it will throw error,since this error is not very little,so we just ignore it and return IOU of 0.
    LOG(ERROR) << e.what();
  }
  */
  return result;
}
template float skew_iou(const Point5f<float> &A, const Point5f<float> &B);
template double skew_iou(const Point5f<double> &A, const Point5f<double> &B);
#ifdef USE_GPU_NMS
template <typename Dtype>
vector<vector<Dtype> > skew_ious(const vector<Point5f<Dtype> > &A, const vector<Point5f<Dtype> > &B) {
  if (A.size() < 100) { // use CPU instead
	vector<vector<Dtype> >ious;
	for (size_t i = 0; i < A.size(); i++) {
		vector<Dtype> sub_ious;
		for (size_t j = 0; j < B.size(); j++) {
			sub_ious.push_back(skew_iou(A[i], B[j]));
		}
		ious.push_back(sub_ious);
	}
	return ious;
  }
    vector<float> bboxes(A.size() * 5);
    vector<float> query_boxes(B.size() * 5);
    for(int i=0;i<A.size();i++) {
        bboxes[i * 5] = A[i][0];
        bboxes[i * 5 + 1] = A[i][1];
        bboxes[i * 5 + 2] = A[i][2];
        bboxes[i * 5 + 3] = A[i][3];
        bboxes[i * 5 + 4] = A[i][4];
    }
    for(int i=0;i<B.size();i++) {
        query_boxes[i * 5] = B[i][0];
        query_boxes[i * 5 + 1] = B[i][1];
        query_boxes[i * 5 + 2] = B[i][2];
        query_boxes[i * 5 + 3] = B[i][3];
        query_boxes[i * 5 + 4] = B[i][4];
    }
    Dtype type = 0;//not used truely,only for template
    //return get_rbbox_ious_gpu(bboxes, query_boxes, type);
    
    // CHECK cpu code and gpu code get same result
    vector<vector<Dtype> >ious_gpu = get_rbbox_ious_gpu(bboxes, query_boxes, type);
    /*
	for (size_t i = 0; i < A.size(); i++) {
		for (size_t j = 0; j < B.size(); j++) {
			Dtype sub_ious = skew_iou(A[i], B[j]);
            //CHECK_LT(fabs(ious_gpu[i][j]-sub_ious),0.01);
            if(fabs(ious_gpu[i][j]-sub_ious) > 0.01) std::cout << "iou gpu <> cpu " << ious_gpu[i][j] << " vs " << sub_ious << " # " << A[i].to_string() << " and " << B[j].to_string() << std::endl;
		}
	}
    */
    return ious_gpu;
}
#else
template <typename Dtype>
vector<vector<Dtype> > skew_ious(const vector<Point5f<Dtype> > &A, const vector<Point5f<Dtype> > &B) {
	vector<vector<Dtype> >ious;
	for (size_t i = 0; i < A.size(); i++) {
		vector<Dtype> sub_ious;
		for (size_t j = 0; j < B.size(); j++) {
			sub_ious.push_back(skew_iou(A[i], B[j]));
		}
		ious.push_back(sub_ious);
	}
	return ious;
}
#endif
template vector<vector<double> > skew_ious(const vector<Point5f<double> > &A, const vector<Point5f<double> > &B);
template vector<vector<float> > skew_ious(const vector<Point5f<float> > &A, const vector<Point5f<float> > &B);

template <typename Dtype>
Dtype get_iou(const Point4f<Dtype> &A, const Point4f<Dtype> &B) {
  const Dtype xx1 = std::max(A[0], B[0]);
  const Dtype yy1 = std::max(A[1], B[1]);
  const Dtype xx2 = std::min(A[2], B[2]);
  const Dtype yy2 = std::min(A[3], B[3]);
  Dtype inter = std::max(Dtype(0), xx2 - xx1 + 1) * std::max(Dtype(0), yy2 - yy1 + 1);
  Dtype areaA = (A[2] - A[0] + 1) * (A[3] - A[1] + 1);
  Dtype areaB = (B[2] - B[0] + 1) * (B[3] - B[1] + 1);
  return inter / (areaA + areaB - inter);
}
template float get_iou(const Point4f<float> &A, const Point4f<float> &B);
template double get_iou(const Point4f<double> &A, const Point4f<double> &B);

template <typename Dtype>
vector<vector<Dtype> > get_ious(const vector<Point4f<Dtype> > &A, const vector<Point4f<Dtype> > &B) {
  vector<vector<Dtype> >ious;
  for (size_t i = 0; i < A.size(); i++) {
    ious.push_back(get_ious(A[i], B));
  }
  return ious;
}
template vector<vector<float> > get_ious(const vector<Point4f<float> > &A, const vector<Point4f<float> > &B);
template vector<vector<double> > get_ious(const vector<Point4f<double> > &A, const vector<Point4f<double> > &B);

template <typename Dtype>
vector<Dtype> get_ious(const Point4f<Dtype> &A, const vector<Point4f<Dtype> > &B) {
  vector<Dtype> ious;
  for (size_t i = 0; i < B.size(); i++) {
    ious.push_back(get_iou(A, B[i]));
  }
  return ious;
}

template vector<float> get_ious(const Point4f<float> &A, const vector<Point4f<float> > &B);
template vector<double> get_ious(const Point4f<double> &A, const vector<Point4f<double> > &B);

//fyk angle differ < 90 degree
template <typename Dtype>
vector<vector<Dtype> > angle_diff(const vector<Point5f<Dtype> > &A, const vector<Point5f<Dtype> > &B) {
  vector<vector<Dtype> >diffs;
  for (size_t i = 0; i < A.size(); i++) {
	vector<Dtype> sub_diffs;
	for (size_t j = 0; j < B.size(); j++) {
		Dtype d = fabs(A[i][4] - B[j][4]);
        while(d > M_PI) d -= M_PI;
		if(d>M_PI_2) d = M_PI - d;
		sub_diffs.push_back(d);
	}
	diffs.push_back(sub_diffs);
  }
  return diffs;
}
template vector<vector<float> > angle_diff(const vector<Point5f<float> > &A, const vector<Point5f<float> > &B);
template vector<vector<double> > angle_diff(const vector<Point5f<double> > &A, const vector<Point5f<double> > &B);

template <typename Dtype>
Point4f<Dtype> rotate_outer_box_coordinates(const Point5f<Dtype> &rbox) {
	//get rotated box's axis-aligned outer rectangle
    /*
	Dtype a  = fabs(rbox[4]);
    Dtype c  = cos(a);
    Dtype s  = sin(a);
	Dtype bw = (rbox[2] * c + rbox[3] * s)/2;
	Dtype bh = (rbox[2] * s + rbox[3] * c)/2;
	Dtype x1 = rbox[0] - bw;
	Dtype x2 = rbox[0] + bw;
	Dtype y1 = rbox[1] - bh;
	Dtype y2 = rbox[1] + bh;
	Point4f<Dtype> rect(x1,y1,x2,y2);
	return rect;
    */
    // our angle(rad) is anti clock-wise
    float a_cos  =  cos(rbox[4]);
    float a_sin  =  sin(rbox[4]);
    float ctr_x = rbox[0];
    float ctr_y = rbox[1];
    float w = rbox[2];
    float h = rbox[3];
    
    float pts[8];
    float pts_x[4];
    float pts_y[4];

    pts_x[0] = - w / 2;
    pts_x[1] = - w / 2;
    pts_x[2] = w / 2;
    pts_x[3] = w / 2;
    
    pts_y[0] = - h / 2;
    pts_y[1] = h / 2;
    pts_y[2] = h / 2;
    pts_y[3] = - h / 2;

    float min_x = INT_MAX, max_x = INT_MIN, min_y = INT_MAX, max_y = INT_MIN;
    for(int i = 0;i < 4;i++) {
        pts[2 * i] = a_cos * pts_x[i] - a_sin * pts_y[i] + ctr_x;
        pts[2 * i + 1] = a_sin * pts_x[i] + a_cos * pts_y[i] + ctr_y;
        if (pts[2 * i] < min_x) min_x = pts[2 * i];
        else if (pts[2 * i] > max_x) max_x = pts[2 * i];
        if (pts[2 * i + 1] < min_y) min_y = pts[2 * i + 1];
        else if (pts[2 * i + 1] > max_y) max_y = pts[2 * i + 1];
    }
    Point4f<Dtype> rect(min_x,min_y,max_x,max_y);
    //CHECK_LT(rect[0],rect[2]) << rect.to_string();
    //CHECK_LT(rect[1],rect[3]) << rect.to_string();
    return rect;
}
template Point4f<float> rotate_outer_box_coordinates(const Point5f<float> &rbox);
template Point4f<double> rotate_outer_box_coordinates(const Point5f<double> &rbox);

float get_scale_factor(int width, int height, int short_size, int max_long_size) {
  float im_size_min = std::min(width, height);
  float im_size_max = std::max(width, height);

  float scale_factor = static_cast<float>(short_size) / im_size_min;
  // Prevent the biggest axis from being more than max_size
  if (scale_factor * im_size_max > max_long_size) {
    scale_factor = static_cast<float>(max_long_size) / im_size_max;
  }
  return scale_factor;
}

} // namespace frcnn

} // namespace caffe
