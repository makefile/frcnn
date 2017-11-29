// ------------------------------------------------------------------
// Xuanyi . Refer to Dong Jian
// 2016/03/30
// ------------------------------------------------------------------
#ifndef CAFFE_FRCNN_ROI_DATA_LAYER_HPP_
#define CAFFE_FRCNN_ROI_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

namespace Frcnn {

/*************************************************
 * FRCNN_ROI_DATA
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
 *   # image_index
 *   img_path (rel path)
 *   num_roi
 *   label x1 y1 x2 y2
 * ........
 * please make sure image_index start from 0 and be continue
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class FrcnnRoiDataLayer : public BasePrefetchingDataLayer<Dtype>  {
 public:
  explicit FrcnnRoiDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~FrcnnRoiDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "FrcnnRoiData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:

  virtual void ShuffleImages();
  virtual unsigned int PrefetchRand();
  virtual void load_batch(Batch<Dtype>* batch);
  virtual void CheckResetRois(vector<vector<float> > &rois, const string image_path, const float cols, const float rows, const float im_scale);
  virtual void FlipRois(vector<vector<float> > &rois, const float cols);

  // Random Seed /if use multigpu set for synchronization
  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector< std::string > image_database_;
  vector<vector<vector<float> > >roi_database_;
  float mean_values_[3];
  // cache_images: will load all images in memory for faster access
  bool cache_images_;
  vector<std::pair<std::string, Datum> > image_database_cache_;
  //
  vector<int> lines_;
  int lines_id_;
  float max_short_;
  float max_long_;
};

}  // namespace Frcnn

}  // namespace caffe

#endif  // CAFFE_FRCNN_ROI_DATA_LAYER_HPP_
