// ------------------------------------------------------------------
// FPN
// Written by github.com/makefile
// ------------------------------------------------------------------
#ifndef FPN_UTILS_HPP_
#define FPN_UTILS_HPP_

#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/blob.hpp"
using namespace caffe;
using namespace caffe::Frcnn;
// single scale version forked from generate_anchors.py
//vector<int> generate_anchors(int base_size=16, vector<float> ratios={0.5, 1, 2}, int scale=8) {
std::vector<std::vector<int> > generate_anchors(int base_size, const std::vector<float> &ratios, const std::vector<int> &scales);

//calc pyramid level of rois
template <typename Dtype>
int calc_level(Point4f<Dtype> &box, int max_level) ;

//put rois to different pyramid level top blob
template <typename Dtype>
void split_top_rois_by_level(const vector<Blob<Dtype> *> &top, int roi_blob_start_idx, std::vector<std::vector<Point4f<Dtype> > > &level_rois);

#endif

