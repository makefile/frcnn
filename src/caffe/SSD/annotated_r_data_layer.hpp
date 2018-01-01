#ifndef CAFFE_ANNOTATED_R_DATA_LAYER_HPP_
#define CAFFE_ANNOTATED_R_DATA_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "data_reader.hpp"
#include "ssd_data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "ssd_base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class AnnotatedRDataLayer : public SSDBasePrefetchingDataLayer<Dtype> {
 public:
  explicit AnnotatedRDataLayer(const LayerParameter& param);
  virtual ~AnnotatedRDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // AnnotatedDataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "AnnotatedData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  DataReader<AnnotatedDatumR> reader_;
  bool has_anno_type_;
  AnnotatedDatumR_AnnotationType anno_type_;
  vector<BatchSampler> batch_samplers_;
  string label_map_file_;
};

}  // namespace caffe

#endif  // CAFFE_ANNOTATED_r_DATA_LAYER_HPP_
