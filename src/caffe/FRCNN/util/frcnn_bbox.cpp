#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_gpu_nms.hpp"

namespace caffe {

namespace Frcnn {

INSTANTIATE_CLASS(Point4f);
INSTANTIATE_CLASS(BBox);

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
Dtype get_iou(const BBox<Dtype> &A, const BBox<Dtype> &B) {
  const Dtype xx1 = std::max(A[0], B[0]);
  const Dtype yy1 = std::max(A[1], B[1]);
  const Dtype xx2 = std::min(A[2], B[2]);
  const Dtype yy2 = std::min(A[3], B[3]);
  Dtype inter = std::max(Dtype(0), xx2 - xx1 + 1) * std::max(Dtype(0), yy2 - yy1 + 1);
  Dtype areaA = (A[2] - A[0] + 1) * (A[3] - A[1] + 1);
  Dtype areaB = (B[2] - B[0] + 1) * (B[3] - B[1] + 1);
  return inter / (areaA + areaB - inter);
}
template float get_iou(const BBox<float> &A, const BBox<float> &B);
template double get_iou(const BBox<double> &A, const BBox<double> &B);

template <typename Dtype>
vector<vector<Dtype> > get_ious(const vector<Point4f<Dtype> > &A, const vector<Point4f<Dtype> > &B, bool use_gpu) {
#ifdef USE_GPU_NMS
  if (use_gpu) {
    vector<float> bboxes(A.size() * 4);
    vector<float> query_boxes(B.size() * 4);
    for(int i=0;i<A.size();i++) {
        bboxes[i * 4] = A[i][0];
        bboxes[i * 4 + 1] = A[i][1];
        bboxes[i * 4 + 2] = A[i][2];
        bboxes[i * 4 + 3] = A[i][3];
    }
    for(int i=0;i<B.size();i++) {
        query_boxes[i * 4] = B[i][0];
        query_boxes[i * 4 + 1] = B[i][1];
        query_boxes[i * 4 + 2] = B[i][2];
        query_boxes[i * 4 + 3] = B[i][3];
    }
    // number of boxes
    int n = bboxes.size() / 4;
    int k = query_boxes.size() / 4;
    std::vector<std::vector<Dtype> > ious(n,std::vector<Dtype>(k));//array of [n*k]
    float *overlaps = new float[n * k];
    _overlaps(overlaps, &bboxes[0], &query_boxes[0],n,k);
    for(int i=0;i<n;i++)
        for(int j=0;j<k;j++)
            ious[i][j] = overlaps[i * k + j];
    delete overlaps;
    return ious;
  } else {
    vector<vector<Dtype> >ious;
    for (size_t i = 0; i < A.size(); i++) {
      ious.push_back(get_ious(A[i], B));
    }
    return ious;
  }
#else
  vector<vector<Dtype> >ious;
  for (size_t i = 0; i < A.size(); i++) {
    ious.push_back(get_ious(A[i], B));
  }
  return ious;
#endif
}
template vector<vector<float> > get_ious(const vector<Point4f<float> > &A, const vector<Point4f<float> > &B, bool use_gpu);
template vector<vector<double> > get_ious(const vector<Point4f<double> > &A, const vector<Point4f<double> > &B, bool use_gpu);

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

template <typename Dtype>
vector<BBox<Dtype> > bbox_vote(const vector<BBox<Dtype> > &dets_NMS, const vector<BBox<Dtype> > &dets_all, Dtype iou_thresh, Dtype add_val) {
    unsigned int N = dets_NMS.size();
    unsigned int M = dets_all.size();
    vector<BBox<Dtype> > dets_voted(N);

    for (size_t i = 0; i < N; i++) {
      Dtype acc_score = 1e-8;
      BBox<Dtype> acc_box;
      for (size_t j = 0; j < M; j++) {
        if (get_iou(dets_NMS[i], dets_all[j]) < iou_thresh)
          continue;
        Dtype score = dets_all[j].confidence + add_val;//[4]
        //score = std::max(Dtype(0), score); // neighbor score
        //acc_box += dets_all[4] * dets_all[0:4]
        acc_box[0] += score * dets_all[j][0];
        acc_box[1] += score * dets_all[j][1];
        acc_box[2] += score * dets_all[j][2];
        acc_box[3] += score * dets_all[j][3];
        acc_score  += score;
      }
      dets_voted[i][0] = acc_box[0] / acc_score;
      dets_voted[i][1] = acc_box[1] / acc_score;
      dets_voted[i][2] = acc_box[2] / acc_score;
      dets_voted[i][3] = acc_box[3] / acc_score;
      dets_voted[i].confidence = dets_NMS[i].confidence; // Keep the original score
      dets_voted[i].id = dets_NMS[i].id; //[5] class id
    }
    return dets_voted;
}

template vector<BBox<float> > bbox_vote(const vector<BBox<float> >& , const vector<BBox<float> >& , float iou_thresh, float add_val);
template vector<BBox<double> > bbox_vote(const vector<BBox<double> >& , const vector<BBox<double> >& , double iou_thresh, double add_val);

} // namespace frcnn

} // namespace caffe
