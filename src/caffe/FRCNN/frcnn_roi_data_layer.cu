#include "caffe/FRCNN/frcnn_roi_data_layer.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"

namespace caffe {

namespace Frcnn {

template <typename Dtype>
void FrcnnRoiDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = this->prefetch_full_.pop("Data layer prefetch queue empty");
  //LOG(INFO) << "====================data layer batch:" << batch->data_.num();//fyk
  //LOG(INFO) << "batch->label_.shape_string: " << batch->label_.shape_string();
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data, Image Blob
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(), top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // top[1] is image_info , top[2] is gx_bpxes
    //caffe_copy(3, batch->label_.gpu_data(), top[1]->mutable_gpu_data());
    // fyk modify for supporting batch > 1
    const int batch_size = FrcnnParam::IMS_PER_BATCH;
    caffe_copy(batch_size * 5, batch->label_.gpu_data(), top[1]->mutable_gpu_data());
    //LOG(INFO) << "height: "<<batch->label_.cpu_data()[0] <<" width: "<<batch->label_.cpu_data()[1];
    // Reshape to loaded labels.
    top[2]->Reshape(batch->label_.num()-batch_size, batch->label_.channels(), batch->label_.height(), batch->label_.width());
    // Copy the labels.
    // First five is image_info
    caffe_copy(batch->label_.count() - batch_size * 5, batch->label_.gpu_data() + batch_size * 5, top[2]->mutable_gpu_data());
    for (int j=0;j<3;j++){
//	vector<int> s = top[j].shapei_;
//	LOG(INFO) << "data shpae " << s[0] << " " << s[1] << " " << s[2] << " " << s[3] ;
	//LOG(INFO) << top[j]->shape_string();
    }
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  this->prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(FrcnnRoiDataLayer);

}  // namespace Frcnn

}  // namespace caffe
