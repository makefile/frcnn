// ------------------------------------------------------------------
// FPN
// Written by github.com/makefile
// ------------------------------------------------------------------

#include "fpn_utils.hpp"
#include <cmath>
#include "caffe/blob.hpp"
#include "caffe/FRCNN/util/frcnn_utils.hpp"
using namespace std;
using namespace caffe;
using namespace caffe::Frcnn;
// single scale version forked from generate_anchors.py
//vector<int> generate_anchors(int base_size=16, vector<float> ratios={0.5, 1, 2}, int scale=8) {
vector<vector<int> > generate_anchors(int base_size, vector<float> &ratios, int scale) {
	vector<vector<int> > anchors (3, vector<int>(4,0));
	int w = base_size;
	int h = base_size;
	float x_ctr = 0.5 * (w - 1);
	float y_ctr = 0.5 * (h - 1);
	w *= scale;
	h *= scale;
	int size = w * h;
	for (int i=0;i<ratios.size();i++){
		float r = ratios[i];
		float size_ratio = size / r ;
		w = (int)(sqrt(size_ratio));
		h = (int)(w * r);
		vector<int> &a = anchors[i];//make sure this is reference instead of copy
		a[0] = x_ctr - 0.5 * (w - 1);
		a[1] = y_ctr - 0.5 * (h - 1); 
		a[2] = x_ctr + 0.5 * (w - 1); 
		a[3] = y_ctr + 0.5 * (h - 1);
	}
	return anchors;
}

// get feature pyramid level,range (2~max_level),for RPN,max_level=6;for RCNN,max_level=5
template <typename Dtype>
int calc_level(Point4f<Dtype> &box, int max_level) {
	// assign rois to level Pk    (P2 ~ P_max_level)
	int w = box[2] - box[0];
	int h = box[3] - box[1];
	//224 is base size of ImageNet
	return min(max_level, max(2, (int)(4 + log2(sqrt(w * h) / 224))));
}
template int calc_level(Point4f<float> &box, int max_level);
template int calc_level(Point4f<double> &box, int max_level);

template <typename Dtype>
void split_top_rois_by_level(const vector<Blob<Dtype> *> &top,vector<Point4f<Dtype> > &rois, int n_level) {
	vector<vector<Point4f<Dtype> > > level_boxes (5,vector<Point4f<Dtype> >());
	//int max_idx = 0;
	//int max_roi_num = 0;
  	for (size_t i = 0; i < rois.size(); i++) {
		int level_idx = calc_level(rois[i], n_level + 1) - 2;
		level_boxes[level_idx].push_back(rois[i]);
		//if(level_boxes[level_idx].size() > max_roi_num){
		//	max_roi_num = level_boxes[level_idx].size();
		//	max_idx = level_idx;
		//}
	}
	//random move 1 roi to empty level_boxes for that blob with num=0 will cause CUDA check error.
	//this method is a little dirty,and if the rois num is less than pyramid levels num,then it is not enough to divide.
	//so I modify CAFFE_GET_BLOCKS in include/caffe/util/device_alternate.hpp instead to support blob count=0.
	/*for (size_t level_idx = 0; level_idx < 5; level_idx++) {
		int num = level_boxes[level_idx].size();
		if(0==num){
			level_boxes[level_idx].push_back(level_boxes[max_idx].back());
			level_boxes[max_idx].pop_back();
		}
	} */
	for (size_t level_idx = 0; level_idx < 5; level_idx++) {
		top[level_idx]->Reshape(level_boxes[level_idx].size(), 5, 1, 1);
		Dtype *top_data = top[level_idx]->mutable_cpu_data();
		for (size_t i = 0; i < level_boxes[level_idx].size(); i++) {
			Point4f<Dtype> &box = level_boxes[level_idx][i];
			top_data[i * 5] = 0;// fyk: image idx
			for (int j = 1; j < 5; j++) {
			  top_data[i * 5 + j] = box[j - 1];
			}
		}
	}
}
template void split_top_rois_by_level(const vector<Blob<float> *> &top,vector<Point4f<float> > &rois, int n_level);
template void split_top_rois_by_level(const vector<Blob<double> *> &top,vector<Point4f<double> > &rois, int n_level);

