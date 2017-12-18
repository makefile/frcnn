#ifndef CAFFE_FRCNN_GPU_NMS_HPP_
#define CAFFE_FRCNN_GPU_NMS_HPP_

namespace caffe {

namespace Frcnn {

void gpu_nms(int* keep_out, int* num_out, const float* boxes_dev, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int device_id=-1);
// gpu nms,load from host to GPU device
void _rotate_nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
                  int boxes_dim, float nms_overlap_thresh, int device_id=0);

} // namespace frcnn

} // namespace caffe
#endif // CAFFE_FRCNN_UTILS_HPP_
