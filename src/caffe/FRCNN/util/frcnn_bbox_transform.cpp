#include "caffe/FRCNN/util/frcnn_utils.hpp"

namespace caffe {

namespace Frcnn {

using std::vector;

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
