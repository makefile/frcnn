// ------------------------------------------------------------------
// Xuanyi . Refer to Dong Jian
// 2016/03/30
// ------------------------------------------------------------------
#ifndef CAFFE_FRCNN_VID_DATA_LAYER_HPP_
#define CAFFE_FRCNN_VID_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

namespace Frcnn {

/*************************************************
 * FRCNN_VID_DATA
 * The data layer used during training to train a Fast R-CNN network.
 * Refer to "RoIDataLayer implements a Caffe Python layer".
 * top: 'data'
 * top: 'im_info'
 * top: 'gt_boxes'
 * @brief Provides data to the Net from windows of images files, specified
 *        by a window data file.
 *
 * Copy From windows_data_layers
 * With Special_Labels
 * First Labels is 0 , indicate is image_info w h w_pad h_pad
 * Follows With labels x1 y1 x2 y2
 *
 * The Lastest ars -1 0 0 0 0 for alignment with batch images!
 *
 * roi_data_file format
 * repeated:
 *   # folder 
 *   num height width
 *   img_path num_roi 
 *   num_roi
 *   track_id label x1 y1 x2 y2
 * ........
 * please make sure image_index start from 0 and be continue
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class FrcnnVidDataLayer : public BasePrefetchingDataLayer<Dtype>  {
 public:
  explicit FrcnnVidDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~FrcnnVidDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "FrcnnVidData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:

  virtual unsigned int PrefetchRand();
  virtual void load_batch(Batch<Dtype>* batch);

  // Random Seed /if use multigpu set for synchronization
  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<VidPrepare<Dtype> > vid_database_;
  float mean_values_[3];
  // cache_images: will load all images in memory for faster access
  bool cache_images_;
  float max_short_;
  float max_long_;
};

}  // namespace Frcnn

}  // namespace caffe

#endif  // CAFFE_FRCNN_VID_DATA_LAYER_HPP_
