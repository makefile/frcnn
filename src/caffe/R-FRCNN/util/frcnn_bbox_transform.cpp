#include "caffe/FRCNN/util/frcnn_utils.hpp"

namespace caffe {

namespace Frcnn {

using std::vector;
//fyk rotated box recover
template <typename Dtype>
Point5f<Dtype> bbox_transform_inv(const Point5f<Dtype>& box, const Point5f<Dtype>& delta) {
  Dtype src_w = box[2] ;
  Dtype src_h = box[3] ;
  Dtype src_ctr_x = box[0] ; // box[0] + 0.5*src_w;
  Dtype src_ctr_y = box[1] ; // box[1] + 0.5*src_h;
  Dtype src_a = box[4];
  Dtype pred_ctr_x = delta[0] * src_w * cos(src_a) -delta[1] * src_h * sin(src_a) + src_ctr_x;
  Dtype pred_ctr_y = delta[0] * src_w * sin(src_a) + delta[1] * src_h * cos(src_a) + src_ctr_y;
  Dtype pred_w = exp(delta[2]) * src_w;
  Dtype pred_h = exp(delta[3]) * src_h;
  Dtype pred_a = delta[4] * M_PI_2 + src_a;//pi/2;
  //while(pred_a <= - M_PI_2) pred_a += M_PI;
  //while(pred_a > M_PI_2) pred_a -= M_PI;
  //CHECK(pred_a > - M_PI_2 && pred_a <= M_PI_2) << pred_a;
  if(pred_a <= - M_PI_2) pred_a += M_PI;
  if(pred_a > M_PI_2) pred_a -= M_PI;

  return Point5f<Dtype>(pred_ctr_x, pred_ctr_y,pred_w, pred_h, pred_a);
}
template Point5f<float> bbox_transform_inv(const Point5f<float>& box, const Point5f<float>& delta);
template Point5f<double> bbox_transform_inv(const Point5f<double>& box, const Point5f<double>& delta);

template <typename Dtype>
Point4f<Dtype> bbox_transform_inv(const Point4f<Dtype>& box, const Point4f<Dtype>& delta) {
  Dtype src_w = box[2] - box[0] + 1;
  Dtype src_h = box[3] - box[1] + 1;
  Dtype src_ctr_x = box[0] + 0.5 * src_w; // box[0] + 0.5*src_w;
  Dtype src_ctr_y = box[1] + 0.5 * src_h; // box[1] + 0.5*src_h;
  Dtype pred_ctr_x = delta[0] * src_w + src_ctr_x;
  Dtype pred_ctr_y = delta[1] * src_h + src_ctr_y;
  Dtype pred_w = exp(delta[2]) * src_w;
  Dtype pred_h = exp(delta[3]) * src_h;
  return Point4f<Dtype>(pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h,
              pred_ctr_x + 0.5 * pred_w, pred_ctr_y + 0.5 * pred_h);
  // return Point4f<Dtype>(pred_ctr_x - 0.5*(pred_w-1) , pred_ctr_y - 0.5*(pred_h-1) ,
  // pred_ctr_x + 0.5*(pred_w-1) , pred_ctr_y + 0.5*(pred_h-1));
}
template Point4f<float> bbox_transform_inv(const Point4f<float>& box, const Point4f<float>& delta);
template Point4f<double> bbox_transform_inv(const Point4f<double>& box, const Point4f<double>& delta);

template <typename Dtype>
vector<Point4f<Dtype> > bbox_transform_inv(const Point4f<Dtype>& box, const vector<Point4f<Dtype> >& deltas) {
  vector<Point4f<Dtype> > ans;
  for (size_t index = 0; index < deltas.size(); index++) {
    ans.push_back(bbox_transform_inv(box, deltas[index]));
  }
  return ans;
}
template vector<Point4f<float> > bbox_transform_inv(const Point4f<float>& box, const vector<Point4f<float> >& deltas);
template vector<Point4f<double> > bbox_transform_inv(const Point4f<double>& box, const vector<Point4f<double> >& deltas);

//fyk implement of box regression target as in paper "Rotated Region based CNN for Ship Detection"
template <typename Dtype>
Point5f<Dtype> bbox_transform(const Point5f<Dtype>& ex_roi, const Point5f<Dtype>& gt_roi) {
  Dtype ex_width = ex_roi[2];
  Dtype ex_height = ex_roi[3];
  Dtype ex_ctr_x = ex_roi[0];
  Dtype ex_ctr_y = ex_roi[1];
  Dtype ex_theta = ex_roi[4];
  Dtype gt_widths = gt_roi[2];
  Dtype gt_heights = gt_roi[3];
  Dtype gt_ctr_x = gt_roi[0];
  Dtype gt_ctr_y = gt_roi[1];
  Dtype gt_theta = gt_roi[4];
  
  //there the gt_theta & ex_theta should all in (-π/2,π/2], when learning is coveraged
  //while(ex_theta <= - M_PI_2) ex_theta += M_PI;
  //while(ex_theta > M_PI_2) ex_theta -= M_PI;

  Dtype ce = cos(ex_theta);
  Dtype se = sin(ex_theta);
  Dtype targets_dx = ( ce * (gt_ctr_x - ex_ctr_x) + se * (gt_ctr_y - ex_ctr_y) ) / ex_width;
  Dtype targets_dy = ( - se * (gt_ctr_x - ex_ctr_x) + ce * (gt_ctr_y - ex_ctr_y) ) / ex_height;
  Dtype targets_dw = log(gt_widths / ex_width);
  Dtype targets_dh = log(gt_heights / ex_height);
  Dtype targets_da = gt_theta - ex_theta;
  //while(targets_da < - M_PI) targets_da += M_PI;
  //while(ex_theta > M_PI) targets_da -= M_PI;
  if(gt_theta <= - M_PI_2 && ex_theta > M_PI_2) targets_da += M_PI;
  if(gt_theta > M_PI_2 && ex_theta <= - M_PI_2) targets_da -= M_PI;
  targets_da *= M_2_PI; // 2/pi 
  //CHECK(gt_theta > - M_PI_2 && gt_theta <= M_PI_2 && ex_theta > - M_PI_2 && ex_theta <= M_PI_2 ) << gt_theta << " " << ex_theta;
  return Point5f<Dtype>(targets_dx, targets_dy, targets_dw, targets_dh, targets_da);
}
template Point5f<float> bbox_transform(const Point5f<float>& ex_roi, const Point5f<float>& gt_roi);
template Point5f<double> bbox_transform(const Point5f<double>& ex_roi, const Point5f<double>& gt_roi);

template <typename Dtype>
vector<Point5f<Dtype> > bbox_transform(const vector<Point5f<Dtype> >& ex_rois, const vector<Point5f<Dtype> >& gt_rois) {
  CHECK_EQ(ex_rois.size(), gt_rois.size());
  vector<Point5f<Dtype> > transformed_bbox;
  for (size_t i = 0; i < gt_rois.size(); i++) {
    transformed_bbox.push_back(bbox_transform(ex_rois[i], gt_rois[i]));
  }
  return transformed_bbox;
}
template vector<Point5f<float> > bbox_transform(const vector<Point5f<float> >& ex_rois, const vector<Point5f<float> >& gt_rois);
template vector<Point5f<double> > bbox_transform(const vector<Point5f<double> >& ex_rois, const vector<Point5f<double> >& gt_rois);

template <typename Dtype>
Point4f<Dtype> bbox_transform(const Point4f<Dtype>& ex_roi, const Point4f<Dtype>& gt_roi) {
  Dtype ex_width = ex_roi[2] - ex_roi[0] + 1;
  Dtype ex_height = ex_roi[3] - ex_roi[1] + 1;
  Dtype ex_ctr_x = ex_roi[0] + 0.5 * ex_width;
  Dtype ex_ctr_y = ex_roi[1] + 0.5 * ex_height;
  Dtype gt_widths = gt_roi[2] - gt_roi[0] + 1;
  Dtype gt_heights = gt_roi[3] - gt_roi[1] + 1;
  Dtype gt_ctr_x = gt_roi[0] + 0.5 * gt_widths;
  Dtype gt_ctr_y = gt_roi[1] + 0.5 * gt_heights;
  Dtype targets_dx = (gt_ctr_x - ex_ctr_x) / ex_width;
  Dtype targets_dy = (gt_ctr_y - ex_ctr_y) / ex_height;
  Dtype targets_dw = log(gt_widths / ex_width);
  Dtype targets_dh = log(gt_heights / ex_height);
  return Point4f<Dtype>(targets_dx, targets_dy, targets_dw, targets_dh);
}
template Point4f<float> bbox_transform(const Point4f<float>& ex_roi, const Point4f<float>& gt_roi);
template Point4f<double> bbox_transform(const Point4f<double>& ex_roi, const Point4f<double>& gt_roi);

template <typename Dtype>
vector<Point4f<Dtype> > bbox_transform(const vector<Point4f<Dtype> >& ex_rois, const vector<Point4f<Dtype> >& gt_rois) {
  CHECK_EQ(ex_rois.size(), gt_rois.size());
  vector<Point4f<Dtype> > transformed_bbox;
  for (size_t i = 0; i < gt_rois.size(); i++) {
    transformed_bbox.push_back(bbox_transform(ex_rois[i], gt_rois[i]));
  }
  return transformed_bbox;
}
template vector<Point4f<float> > bbox_transform(const vector<Point4f<float> >& ex_rois, const vector<Point4f<float> >& gt_rois);
template vector<Point4f<double> > bbox_transform(const vector<Point4f<double> >& ex_rois, const vector<Point4f<double> >& gt_rois);

} // namespace frcnn

} // namespace caffe
