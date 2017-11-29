#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/ACTION_REC/video_data_layer.hpp"

namespace caffe{

template <typename Dtype>
VideoDataLayer<Dtype>:: ~VideoDataLayer<Dtype>(){
	this->StopInternalThread();  
}

template <typename Dtype>
void VideoDataLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
    const int new_height  = this->layer_param_.video_data_param().new_height();
    const int new_width  = this->layer_param_.video_data_param().new_width();
    const int new_length  = this->layer_param_.video_data_param().new_length();
    const int num_segments = this->layer_param_.video_data_param().num_segments();
    CHECK_GT( num_segments , 0 );
    const string& source = this->layer_param_.video_data_param().source();

    LOG(INFO) << "Opening file: " << source;
    std:: ifstream infile(source.c_str());
    string filename;
    int label;
    int length;
    int zero_count = 0;
    while (infile >> filename >> length >> label){
        if (length < new_length){
            zero_count ++;
            continue;
        }
        lines_.push_back(std::make_pair(filename,label));
        lines_duration_.push_back(length);
    }
    if (this->layer_param_.video_data_param().shuffle()){
        const unsigned int prefectch_rng_seed = caffe_rng_rand();
        prefetch_rng_1_.reset(new Caffe::RNG(prefectch_rng_seed));
        prefetch_rng_2_.reset(new Caffe::RNG(prefectch_rng_seed));
        ShuffleVideos();
    }

    LOG(INFO) << "A total of " << lines_.size() << " videos.  zeros frames videos " << zero_count;
    lines_id_ = 0;

    Datum datum;
    const unsigned int frame_prefectch_rng_seed = caffe_rng_rand();
    frame_prefetch_rng_.reset(new Caffe::RNG(frame_prefectch_rng_seed));
    int average_duration = (int) lines_duration_[lines_id_]/num_segments;
    vector<int> offsets;
    for (int i = 0; i < num_segments; ++i){
        caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
        int offset = (*frame_rng)() % (average_duration - new_length + 1);
        offsets.push_back(offset+i*average_duration);
    }
    if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FLOW)
        CHECK(ReadSegmentFlowToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum));
    else
        CHECK(ReadSegmentRGBToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum, true));
    const int crop_size = this->layer_param_.transform_param().crop_size();
    const int batch_size = this->layer_param_.video_data_param().batch_size();
    if (crop_size > 0){
        top[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
        //this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size, crop_size);
        for (int i = 0; i < this->prefetch_.size(); ++i) {
            this->prefetch_[i]->data_.Reshape(batch_size, datum.channels(), crop_size, crop_size);
        }
    } else {
        top[0]->Reshape(batch_size, datum.channels(), datum.height(), datum.width());
        //this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(), datum.width());
        for (int i = 0; i < this->prefetch_.size(); ++i) {
            this->prefetch_[i]->data_.Reshape(batch_size, datum.channels(), datum.height(), datum.width());
        }
    }
    LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();

    top[1]->Reshape(batch_size, 1, 1, 1);
    for(int i = 0 ; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->label_.Reshape(batch_size, 1, 1, 1);
    }

    vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
    this->transformed_data_.Reshape(top_shape);
}

template <typename Dtype>
void VideoDataLayer<Dtype>::ShuffleVideos(){
    caffe::rng_t* prefetch_rng1 = static_cast<caffe::rng_t*>(prefetch_rng_1_->generator());
    caffe::rng_t* prefetch_rng2 = static_cast<caffe::rng_t*>(prefetch_rng_2_->generator());
    shuffle(lines_.begin(), lines_.end(), prefetch_rng1);
    shuffle(lines_duration_.begin(), lines_duration_.end(),prefetch_rng2);
}

template <typename Dtype>
void VideoDataLayer<Dtype>::load_batch(Batch<Dtype>* batch){
    Datum datum;
    CHECK(batch->data_.count());
    Dtype* top_data = batch->data_.mutable_cpu_data();
    Dtype* top_label = batch->label_.mutable_cpu_data();
    VideoDataParameter video_data_param = this->layer_param_.video_data_param();
    const int batch_size = video_data_param.batch_size();
    const int new_height = video_data_param.new_height();
    const int new_width = video_data_param.new_width();
    const int new_length = video_data_param.new_length();
    const int num_segments = video_data_param.num_segments();
    const int lines_size = lines_.size();

    for (int item_id = 0; item_id < batch_size; ++item_id){
        CHECK_GT(lines_size, lines_id_);
        vector<int> offsets;
        int average_duration = (int) lines_duration_[lines_id_] / num_segments;
        while (average_duration == 0){
            LOG(WARNING) << "VideoData : load_batch [] " << item_id << " / " << batch_size << "  .. num_segments : " << num_segments << ",  ave : " << average_duration << " = " << lines_[lines_id_].first;
            Next_Line_Id();
            average_duration = (int) lines_duration_[lines_id_] / num_segments;
        }
        for (int i = 0; i < num_segments; ++i){
            if (this->phase_==TRAIN){
                caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
                int offset = (*frame_rng)() % (average_duration - new_length + 1);
                //LOG(ERROR) << "VideoData : load_batch [] offset : " << offset << ", average_duration : " << average_duration << ", " << new_length;
                offsets.push_back(offset+i*average_duration);
            } else{
                offsets.push_back(int((average_duration-new_length+1)/2 + i*average_duration));
            }
        }
        //LOG(ERROR) << "VideoData : " << (this->phase_==TRAIN?"TRAIN":"TEST") << "　 ---After num_seg";
        if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FLOW){
            if(!ReadSegmentFlowToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum)) {
                LOG(WARNING) << "Load Flow Data Failed, " << item_id << "/" << batch_size << " batch : ( " << lines_[lines_id_].first << " ) [" << lines_[lines_id_].second << "]" 
                    << std::endl << "    Num_of_Segment(" << offsets.size() << ") " << offsets[0] << " ::: Length: " << new_length ;
                item_id --;
                continue;
            }
        } else{
            if(!ReadSegmentRGBToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum, true)) {
                LOG(WARNING) << "Load RGB Data Failed, " << item_id << "/" << batch_size << " batch : ( " << lines_[lines_id_].first << " ) [" << lines_[lines_id_].second << "]" 
                    << std::endl << "    Num_of_Segment(" << offsets.size() << ") " << offsets[0] << " ::: Length: " << new_length ;
                item_id --;
                continue;
            }
		}

        int offset1 = batch->data_.offset(item_id);
    	this->transformed_data_.set_cpu_data(top_data + offset1);
        this->data_transformer_->Transform(datum, &(this->transformed_data_));
        //LOG(ERROR) << "Transform : After";
        top_label[item_id] = lines_[lines_id_].second;

        //next iteration
        Next_Line_Id();
    }
}

template <typename Dtype>
void VideoDataLayer<Dtype>::Next_Line_Id(){
    lines_id_++;
    const int lines_size = lines_.size();
    if (lines_id_ >= lines_size) {
        DLOG(INFO) << "Restarting data prefetching from start.";
        lines_id_ = 0;
        if(this->layer_param_.video_data_param().shuffle()){
            ShuffleVideos();
        }
    }
}

template <typename Dtype>
bool VideoDataLayer<Dtype>::ReadSegmentRGBToDatum(const string& filename, const int label,
    const vector<int> offsets, const int height, const int width, const int length, Datum* datum, bool is_color){
    cv::Mat cv_img;
    string* datum_string;
    char tmp[30];
    int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
        CV_LOAD_IMAGE_GRAYSCALE);
    for (int i = 0; i < offsets.size(); ++i){
        int offset = offsets[i];
        for (int file_id = 1; file_id < length+1; ++file_id){
            sprintf(tmp,"image_%08d.jpg",int(file_id+offset-1));
            string filename_t = filename + "/" + tmp;
            cv::Mat cv_img_origin = cv::imread(filename_t, cv_read_flag);
            if (!cv_img_origin.data){
                LOG(ERROR) << "Could not load file " << filename << std::endl
                    << filename_t << "  {}  " << offset << ", " << file_id << " length : " << length;
                return false;
            }
            if (height > 0 && width > 0){
                cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
            }else{
                cv_img = cv_img_origin;
            }
            int num_channels = (is_color ? 3 : 1);
            if (file_id==1 && i==0){
                datum->set_channels(num_channels*length*offsets.size());
                datum->set_height(cv_img.rows);
                datum->set_width(cv_img.cols);
                datum->set_label(label);
                datum->clear_data();
                datum->clear_float_data();
                datum_string = datum->mutable_data();
            }
            if (is_color) {
                for (int c = 0; c < num_channels; ++c) {
                  for (int h = 0; h < cv_img.rows; ++h) {
                    for (int w = 0; w < cv_img.cols; ++w) {
                      datum_string->push_back(
                        static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
                    }
                  }
                }
              } else {  // Faster than repeatedly testing is_color for each pixel w/i loop
                for (int h = 0; h < cv_img.rows; ++h) {
                  for (int w = 0; w < cv_img.cols; ++w) {
                    datum_string->push_back(
                      static_cast<char>(cv_img.at<uchar>(h, w)));
                    }
                  }
              }
        }
    }
    return true;
}

template <typename Dtype>
bool VideoDataLayer<Dtype>::ReadSegmentFlowToDatum(const string& filename, const int label,
    const vector<int> offsets, const int height, const int width, const int length, Datum* datum){
    cv::Mat cv_img_x, cv_img_y;
    string* datum_string;
    char tmp[30];
    for (int i = 0; i < offsets.size(); ++i){
        int offset = offsets[i];
        for (int file_id = 1; file_id < length+1; ++file_id){
            sprintf(tmp,"flow_x_%08d.jpg",int(file_id+offset));
            string filename_x = filename + "/" + tmp;
            cv::Mat cv_img_origin_x = cv::imread(filename_x, CV_LOAD_IMAGE_GRAYSCALE);
            sprintf(tmp,"flow_y_%08d.jpg",int(file_id+offset));
            string filename_y = filename + "/" + tmp;
            cv::Mat cv_img_origin_y = cv::imread(filename_y, CV_LOAD_IMAGE_GRAYSCALE);
            if (!cv_img_origin_x.data || !cv_img_origin_y.data){
                LOG(ERROR) << "Could not load file " << filename_x << " or " << filename_y;
                return false;
            }
            if (height > 0 && width > 0){
                cv::resize(cv_img_origin_x, cv_img_x, cv::Size(width, height));
                cv::resize(cv_img_origin_y, cv_img_y, cv::Size(width, height));
            }else{
                cv_img_x = cv_img_origin_x;
                cv_img_y = cv_img_origin_y;
            }
            if (file_id==1 && i==0){
                int num_channels = 2;
                datum->set_channels(num_channels*length*offsets.size());
                datum->set_height(cv_img_x.rows);
                datum->set_width(cv_img_x.cols);
                datum->set_label(label);
                datum->clear_data();
                datum->clear_float_data();
                datum_string = datum->mutable_data();
            }
            for (int h = 0; h < cv_img_x.rows; ++h){
                for (int w = 0; w < cv_img_x.cols; ++w){
                    datum_string->push_back(static_cast<char>(cv_img_x.at<uchar>(h,w)));
                }
            }
            for (int h = 0; h < cv_img_y.rows; ++h){
                for (int w = 0; w < cv_img_y.cols; ++w){
                    datum_string->push_back(static_cast<char>(cv_img_y.at<uchar>(h,w)));
                }
            }
        }
    }
    return true;
}

INSTANTIATE_CLASS(VideoDataLayer);
REGISTER_LAYER_CLASS(VideoData);
}
