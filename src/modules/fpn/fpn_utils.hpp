// ------------------------------------------------------------------
// FPN
// Written by github.com/makefile
// ------------------------------------------------------------------
#ifndef FPN_UTILS_HPP_
#define FPN_UTILS_HPP_

#include <cmath>
#include "caffe/blob.hpp"
#include "caffe/FRCNN/util/frcnn_utils.hpp"
using namespace std;
using namespace caffe;
using namespace caffe::Frcnn;
// single scale version forked from generate_anchors.py
//vector<int> generate_anchors(int base_size=16, vector<float> ratios={0.5, 1, 2}, int scale=8) {
vector<vector<int> > generate_anchors(int base_size, vector<float> &ratios, int scale);

//calc pyramid level of rois
template <typename Dtype>
int calc_level(Point4f<Dtype> &box, int max_level) ;

//put rois to different pyramid level top blob
template <typename Dtype>
void split_top_rois_by_level(const vector<Blob<Dtype> *> &top,vector<Point4f<Dtype> > &rois, int n_level) ;

#endif

