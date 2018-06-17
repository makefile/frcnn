// ------------------------------------------------------------------
// FPN
// Written by github.com/makefile
// ------------------------------------------------------------------

#include "fpn_utils.hpp"
#include <cmath>
#include "caffe/blob.hpp"
#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"

using namespace std;
using namespace caffe;
using namespace caffe::Frcnn;
// single scale version forked from generate_anchors.py
//vector<int> generate_anchors(int base_size=16, vector<float> ratios={0.5, 1, 2}, int scale=8) {
vector<vector<int> > generate_anchors(int base_size, const vector<float> &ratios, const vector<int> &scales) {
	vector<vector<int> > anchors (scales.size() * ratios.size(), vector<int>(4,0));
	int w = base_size;
	int h = base_size;
	float x_ctr = 0.5 * (w - 1);
	float y_ctr = 0.5 * (h - 1);
	for (int j=0; j<scales.size(); j++) { 
        const int scale = scales[j];
        w *= scale;
    	h *= scale;
    	int size = w * h;
    	for (int i=0;i<ratios.size();i++){
    		float r = ratios[i];
    		float size_ratio = size / r ;
    		w = (int)(sqrt(size_ratio));
    		h = (int)(w * r);
    		vector<int> &a = anchors[j * ratios.size() + i];//make sure this is reference instead of copy
    		a[0] = x_ctr - 0.5 * (w - 1);
    		a[1] = y_ctr - 0.5 * (h - 1); 
    		a[2] = x_ctr + 0.5 * (w - 1); 
    		a[3] = y_ctr + 0.5 * (h - 1);
    	}
    }
	return anchors;
}

// get feature pyramid level,range (2~max_level),for RPN,max_level=6;for RCNN,max_level=5
template <typename Dtype>
int calc_level(Point4f<Dtype> &box, int max_level) {
	// assign rois to level Pk    (P2 ~ P_max_level)
	Dtype w = box[2] - box[0];
	Dtype h = box[3] - box[1];
	//224 is base size of ImageNet
	//return min(max_level, max(2, (int)(4 + log2(sqrt(w * h) / 224 + 1e-6))));
	return min(max_level, max(2, (int)(FrcnnParam::roi_canonical_level + log2(sqrt(w * h) / FrcnnParam::roi_canonical_scale + 1e-6))));
}
template int calc_level(Point4f<float> &box, int max_level);
template int calc_level(Point4f<double> &box, int max_level);

template <typename Dtype>
void split_top_rois_by_level(const vector<Blob<Dtype> *> &top, int roi_blob_start_idx, vector<vector<Point4f<Dtype> > > &level_rois) {
	//so I modify CAFFE_GET_BLOCKS in include/caffe/util/device_alternate.hpp instead to support blob count=0. !! I am not sure whether this method will cause problem. so, better to add one roi, whether 0 dummy roi or random one
        int n_level = level_rois.size();
	for (size_t level_idx = 0; level_idx < n_level; level_idx++) {
                CHECK(level_rois[level_idx].size() > 0);
                // top 0 may be all rois in proposal layer
		top[roi_blob_start_idx + level_idx]->Reshape(level_rois[level_idx].size(), 5, 1, 1);
		Dtype *top_data = top[roi_blob_start_idx + level_idx]->mutable_cpu_data();
		for (size_t i = 0; i < level_rois[level_idx].size(); i++) {
			Point4f<Dtype> &box = level_rois[level_idx][i];
			top_data[i * 5] = 0;// fyk: image idx
			for (int j = 1; j < 5; j++) {
			  top_data[i * 5 + j] = box[j - 1];
			}
		}
	}
}
template void split_top_rois_by_level(const vector<Blob<float> *> &top, int roi_blob_start_idx, vector<vector<Point4f<float> > > &level_rois);
template void split_top_rois_by_level(const vector<Blob<double> *> &top, int roi_blob_start_idx, vector<vector<Point4f<double> > > &level_rois);

