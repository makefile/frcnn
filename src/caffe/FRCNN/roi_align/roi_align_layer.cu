#include <algorithm>
#include <cfloat>
#include <vector>

#include "roi_align_layer.hpp"


using std::max;
using std::min;

namespace caffe {

	inline __device__ double cubic_coeff_gpu(double x) {
		x = (x>0) ? x : -x;
		if (x<1){
			return 1 - 2 * x*x + x*x*x;
		}
		else if (x<2){
			return 4 - 8 * x + 5 * x*x - x*x*x;
		}
		return 0;
	}

	template <typename Dtype>
	__global__ void ROIAlignForward(const int nthreads, const Dtype* bottom_data,
		const Dtype spatial_scale, const int channels, const int height,
		const int width, const int pooled_height, const int pooled_width,
		const Dtype pad_ratio, const Dtype* bottom_rois, const int interpolate_times, Dtype* top_data, int* argmax_data, Dtype* w_data) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			// (n, c, ph, pw) is an element in the pooled output
			int pw = index % pooled_width;
			int ph = (index / pooled_width) % pooled_height;
			int c = (index / pooled_width / pooled_height) % channels;
			int n = index / pooled_width / pooled_height / channels;

			bottom_rois += n * 5;
			int roi_batch_ind = bottom_rois[0];

			// padding
			Dtype pad_w, pad_h;
			pad_w = (bottom_rois[3] - bottom_rois[1] + 1)*pad_ratio;
			pad_h = (bottom_rois[4] - bottom_rois[2] + 1)*pad_ratio;
			Dtype roi_start_w = (bottom_rois[1] - pad_w) * spatial_scale;
			Dtype roi_start_h = (bottom_rois[2] - pad_h) * spatial_scale;
			Dtype roi_end_w = (bottom_rois[3] + pad_w) * spatial_scale;
			Dtype roi_end_h = (bottom_rois[4] + pad_h) * spatial_scale;
			// clipping
			roi_start_w = max(roi_start_w, Dtype(0)); roi_start_h = max(roi_start_h, Dtype(0));
			int img_width = round(width / spatial_scale);
			int img_height = round(height / spatial_scale);
			roi_end_w = min(Dtype(img_width - 1), roi_end_w);
			roi_end_h = min(Dtype(img_height - 1), roi_end_h);

			Dtype roi_height = max(roi_end_h - roi_start_h + 1, Dtype(1));
			Dtype roi_width = max(roi_end_w - roi_start_w + 1, Dtype(1));
			const Dtype bin_size_h = static_cast<Dtype>(roi_height)
				/ static_cast<Dtype>(pooled_height);
			const Dtype bin_size_w = static_cast<Dtype>(roi_width)
				/ static_cast<Dtype>(pooled_width);

			bottom_data += (roi_batch_ind * channels + c) * height * width;

			double argmax_temp_data[4];
			double w_temp_data[4];
			double start_x = 0.25, start_y = 0.25;
			if (interpolate_times == 1) {
				start_x = 0.5;
				start_y = 0.5;
			}
			Dtype dfValue = 0, maxValue = 0;
			for (int inter_index = 0; inter_index < interpolate_times; ++inter_index) {
				int index_x = inter_index / 2;
				int index_y = inter_index % 2;
				Dtype off_x = index_x * 0.5 + start_x;
				Dtype off_y = index_y * 0.5 + start_y;
				Dtype hcenter = static_cast<Dtype>(ph + off_x)* bin_size_h;
				Dtype wcenter = static_cast<Dtype>(pw + off_y)* bin_size_w;

				hcenter = min(max(hcenter + roi_start_h, Dtype(0)), Dtype(height - 1));
				wcenter = min(max(wcenter + roi_start_w, Dtype(0)), Dtype(width - 1));

				int hstart = min(max(hcenter, Dtype(0)), Dtype(height - 1));
				int wstart = min(max(wcenter, Dtype(0)), Dtype(width - 1));
				int hend = min(max(hstart + 1, 0), height - 1);
				int wend = min(max(wstart + 1, 0), width - 1);

				Dtype fX0 = wcenter - wstart;
				Dtype fX1 = wend - wcenter;
				Dtype fY0 = hcenter - hstart;
				Dtype fY1 = hend - hcenter;
				Dtype fFactorA = fY1 * fX1;
				Dtype fFactorB = fY1 * fX0;
				Dtype fFactorC = fY0 * fX1;
				Dtype fFactorD = fY0 * fX0;

				dfValue = bottom_data[hstart * width + wstart] * fFactorA
					+ bottom_data[hstart * width + wend] * fFactorB
					+ bottom_data[hend * width + wstart] * fFactorC
					+ bottom_data[hend * width + wend] * fFactorD;

				if (inter_index == 0) {
					maxValue = dfValue - 1;
				}

				argmax_temp_data[0] = hstart * width + wstart;
				argmax_temp_data[1] = hstart * width + wend;
				argmax_temp_data[2] = hend * width + wstart;
				argmax_temp_data[3] = hend * width + wend;

				w_temp_data[0] = fFactorA;
				w_temp_data[1] = fFactorB;
				w_temp_data[2] = fFactorC;
				w_temp_data[3] = fFactorD;

				if (dfValue > maxValue || inter_index == 0) {
					maxValue = dfValue;
					top_data[index] = dfValue;
					for (int s = 0; s < 4; ++s) {
						w_data[4 * index + s] = w_temp_data[s];
						argmax_data[4 * index + s] = argmax_temp_data[s];
					}
				}
			}
		}
	}

	template <typename Dtype>
	__global__ void ROICubicForward(const int nthreads, const Dtype* bottom_data,
		const Dtype spatial_scale, const int channels, const int height,
		const int width, const int pooled_height, const int pooled_width,
		const Dtype pad_ratio, const Dtype* bottom_rois, const int interpolate_times, Dtype* top_data, int* argmax_data, Dtype* w_data) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			// (n, c, ph, pw) is an element in the pooled output
			int pw = index % pooled_width;
			int ph = (index / pooled_width) % pooled_height;
			int c = (index / pooled_width / pooled_height) % channels;
			int n = index / pooled_width / pooled_height / channels;

			bottom_rois += n * 5;
			int roi_batch_ind = bottom_rois[0];

			// padding
			Dtype pad_w, pad_h;
			pad_w = (bottom_rois[3] - bottom_rois[1] + 1)*pad_ratio;
			pad_h = (bottom_rois[4] - bottom_rois[2] + 1)*pad_ratio;
			Dtype roi_start_w = (bottom_rois[1] - pad_w) * spatial_scale;
			Dtype roi_start_h = (bottom_rois[2] - pad_h) * spatial_scale;
			Dtype roi_end_w = (bottom_rois[3] + pad_w) * spatial_scale;
			Dtype roi_end_h = (bottom_rois[4] + pad_h) * spatial_scale;
			// clipping
			roi_start_w = max(roi_start_w, Dtype(0)); roi_start_h = max(roi_start_h, Dtype(0));
			int img_width = round(width / spatial_scale);
			int img_height = round(height / spatial_scale);
			roi_end_w = min(Dtype(img_width - 1), roi_end_w);
			roi_end_h = min(Dtype(img_height - 1), roi_end_h);

			Dtype roi_height = max(roi_end_h - roi_start_h + 1, Dtype(1));
			Dtype roi_width = max(roi_end_w - roi_start_w + 1, Dtype(1));
			const Dtype bin_size_h = static_cast<Dtype>(roi_height)
				/ static_cast<Dtype>(pooled_height);
			const Dtype bin_size_w = static_cast<Dtype>(roi_width)
				/ static_cast<Dtype>(pooled_width);

			bottom_data += (roi_batch_ind * channels + c) * height * width;
			double argmax_temp_data[16];
			double w_temp_data[16];
			double start_x = 0.25, start_y = 0.25;
			if (interpolate_times == 1) {
				start_x = 0.5;
				start_y = 0.5;
			}
			Dtype dfCubicValue = 0, maxValue = 0;
			for (int inter_index = 0; inter_index < interpolate_times; ++inter_index) {
				int index_x = inter_index / 2;
				int index_y = inter_index % 2;
				Dtype off_x = index_x * 0.5 + start_x;
				Dtype off_y = index_y * 0.5 + start_y;
				Dtype hcenter = static_cast<Dtype>(ph + off_x)* bin_size_h;
				Dtype wcenter = static_cast<Dtype>(pw + off_y)* bin_size_w;

				hcenter = min(max(hcenter + roi_start_h, Dtype(0)), Dtype(height - 1));
				wcenter = min(max(wcenter + roi_start_w, Dtype(0)), Dtype(width - 1));

				int i = wcenter;
				int j = hcenter;

				/*get adjacent 16 values*/
				double values[4][4];
				int temp_c, temp_r;
				for (int r = j - 1, s = 0; r <= j + 2; r++, s++){
					for (int c = i - 1, t = 0; c <= i + 2; c++, t++){
						//todo: ??16?,????
						temp_c = min(max(Dtype(c), Dtype(0)), Dtype(width - 1));
						temp_r = min(max(Dtype(r), Dtype(0)), Dtype(height - 1));
						values[s][t] = bottom_data[temp_r*width + temp_c];
						argmax_temp_data[s * 4 + t] = temp_r*width + temp_c;
					}
				}

				/*calc the coeff*/
				double u = wcenter - i;
				double v = hcenter - j;
				double A[4], C[4];
				for (int distance = 1, s = 0; distance >= -2; distance--, s++){
					A[s] = cubic_coeff_gpu(u + distance);
					C[s] = cubic_coeff_gpu(v + distance);
				}

				dfCubicValue = 0;
				for (int s = 0; s < 4; s++) {
					for (int t = 0; t < 4; t++) {
						dfCubicValue += values[s][t] * A[t] * C[s];
						w_temp_data[s * 4 + t] = A[t] * C[s];
					}
				}
				if (dfCubicValue > maxValue || inter_index == 0) {
					maxValue = dfCubicValue;
					top_data[index] = dfCubicValue;
					for (int s = 0; s < 16; ++s) {
						w_data[16 * index + s] = w_temp_data[s];
						argmax_data[16 * index + s] = argmax_temp_data[s];
					}
				}
			}
		}
	}

	template <typename Dtype>
	void ROIAlignLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* bottom_rois = bottom[1]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		int* argmax_data = bili_idx.mutable_gpu_data();
		Dtype* w_data = bili_w.mutable_gpu_data();
		int count = top[0]->count();
		int interpolate_times = is_multi_interpolate ? 4 : 1;
		// NOLINT_NEXT_LINE(whitespace/operators)
		if (bi_type == BiCubic) {
			ROICubicForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, bottom_data, spatial_scale_, channels_, height_, width_,
				pooled_height_, pooled_width_, pad_ratio_, bottom_rois, interpolate_times, top_data, argmax_data, w_data);
		}
		else {
			ROIAlignForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, bottom_data, spatial_scale_, channels_, height_, width_,
				pooled_height_, pooled_width_, pad_ratio_, bottom_rois, interpolate_times, top_data, argmax_data, w_data);
		}
		CUDA_POST_KERNEL_CHECK;
	}

	template <typename Dtype>
	__global__ void ROIAlignBackward(const int nthreads, const Dtype* top_diff,
		const int* argmax_data, const Dtype* w_data, const int num_rois, const Dtype spatial_scale,
		const int channels, const int height, const int width,
		const int pooled_height, const int pooled_width, const int w_num, const Dtype pad_ratio,
		Dtype* bottom_diff, const Dtype* bottom_rois) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			// (n, c, h, w) coords in bottom data
			//int pw = index % pooled_width;
			//int ph = (index / pooled_width) % pooled_height;
			int c = (index / pooled_width / pooled_height) % channels;
			int n = index / pooled_width / pooled_height / channels;

			bottom_rois += n * 5;
			int roi_batch_ind = bottom_rois[0];

			for (int i = 0; i < w_num; ++i) {
				if (argmax_data[w_num * index + i] >= 0) {
					int offset_bottom = (roi_batch_ind * channels + c) * height
						* width + argmax_data[w_num * index + i];
					bottom_diff[offset_bottom] += top_diff[index] * w_data[w_num * index + i];
				}
			}
		}
	}

	template <typename Dtype>
	void ROIAlignLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (!propagate_down[0]) {
			return;
		}
		const Dtype* bottom_rois = bottom[1]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const int count = bottom[0]->count();
		caffe_gpu_set(count, Dtype(0.), bottom_diff);
		const int* argmax_data = bili_idx.gpu_data();
		const Dtype* w_data = bili_w.gpu_data();
		const int top_count = top[0]->count();
		int w_num = 4;
		if (bi_type == BiCubic) {
			w_num = 16;
		}
		// NOLINT_NEXT_LINE(whitespace/operators)
		ROIAlignBackward<Dtype> << <CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS >> >(
			top_count, top_diff, argmax_data, w_data, top[0]->num(), spatial_scale_, channels_,
			height_, width_, pooled_height_, pooled_width_, w_num, pad_ratio_, bottom_diff, bottom_rois);
		CUDA_POST_KERNEL_CHECK;
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ROIAlignLayer);

}  // namespace caffe
