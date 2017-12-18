#ifndef _RBBOX_OVERLAPS_HPP_
#define _RBBOX_OVERLAPS_HPP_
#include <vector>
using namespace std;

void _overlaps(float* overlaps,const float* boxes,const float* query_boxes, int n, int k, int device_id);

template <typename Dtype>
vector<vector<Dtype> > get_rbbox_ious_gpu(const vector<float> &boxes, const vector<float> &query_boxes, Dtype type, int device_id=0);


#endif
