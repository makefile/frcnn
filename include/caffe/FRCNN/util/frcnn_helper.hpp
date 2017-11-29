// ------------------------------------------------------------------
// Xuanyi . Refer to Dong Jian
// 2016/04/01
// ------------------------------------------------------------------
#ifndef CAFFE_FRCNN_HELPER_HPP_
#define CAFFE_FRCNN_HELPER_HPP_

#include "caffe/FRCNN/util/frcnn_utils.hpp"

namespace caffe {

namespace Frcnn {

template <typename Dtype>
Point4f<Dtype> bbox_transform(const Point4f<Dtype>& ex_rois,const Point4f<Dtype>& gt_rois);

template <typename Dtype>
std::vector<Point4f<Dtype> > bbox_transform(const std::vector<Point4f<Dtype> >& ex_rois,
                                 const std::vector<Point4f<Dtype> >& gt_rois);

template <typename Dtype>
Point4f<Dtype> bbox_transform_inv(const Point4f<Dtype>& box, const Point4f<Dtype>& delta);

template <typename Dtype>
std::vector<Point4f<Dtype> > bbox_transform_inv(const Point4f<Dtype>& box,
                                      const std::vector<Point4f<Dtype> >& deltas);

}  // namespace frcnn

}  // namespace caffe

#endif
