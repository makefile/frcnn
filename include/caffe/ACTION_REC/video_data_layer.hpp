#ifndef CAFFE_VIDEO_DATA_LAYER_HPP_
#define CAFFE_VIDEO_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/common.hpp"
#include "caffe/util/blocking_queue.hpp"   
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "opencv2/core/version.hpp"

#if CV_MAJOR_VERSION == 2
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

#elif CV_MAJOR_VERSION == 3
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#endif

namespace caffe {

/**
 * @brief Provides data to the Net from video files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class VideoDataLayer : public BasePrefetchingDataLayer<Dtype> {
public:
    explicit VideoDataLayer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param) {}
    virtual ~VideoDataLayer();
    virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "VideoData"; }
    virtual inline int ExactNumBottomBlobs() const { return 0; }
    virtual inline int ExactNumTopBlobs() const { return 2; }

protected:
    shared_ptr<Caffe::RNG> prefetch_rng_;
    shared_ptr<Caffe::RNG> prefetch_rng_2_;
    shared_ptr<Caffe::RNG> prefetch_rng_1_;
    shared_ptr<Caffe::RNG> frame_prefetch_rng_;
    virtual void ShuffleVideos();
    virtual void load_batch(Batch<Dtype>* batch);
    vector<std::pair<std::string, int> > lines_;
    vector<int> lines_duration_;
    int lines_id_;

    // For Video Data Layer to Handle Data
    bool ReadSegmentFlowToDatum(const string& filename, const int label,
        const vector<int> offsets, const int height, const int width, const int length, Datum* datum);

    bool ReadSegmentRGBToDatum(const string& filename, const int label,
        const vector<int> offsets, const int height, const int width, const int length, Datum* datum, bool is_color);

    void Next_Line_Id();
};

}  // namespace caffe

#endif  // CAFFE_VIDEO_DATA_LAYER_HPP_
