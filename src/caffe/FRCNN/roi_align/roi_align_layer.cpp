#include <algorithm>
#include <cfloat>
#include <vector>

#include "roi_align_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

	template <typename Dtype>
	double ROIAlignLayer<Dtype>::cubic_coeff(double x){
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
	double ROIAlignLayer<Dtype>::ResampleCubic(double x, double y, const Dtype * pdfValue, int nWidth, int nHeight, int pool_index, int* argmax_data, Dtype* w_data)
	{
		Dtype dfCubicValue;
		int i = x;
		int j = y;
		/*get adjacent 16 values*/
		double values[4][4];
		int temp_c, temp_r;
		for (int r = j - 1, s = 0; r <= j + 2; r++, s++){
			for (int c = i - 1, t = 0; c <= i + 2; c++, t++){
				//todo: 判断16次，移出循环
				temp_c = min(max(Dtype(c), Dtype(0)), Dtype(nWidth - 1));
				temp_r = min(max(Dtype(r), Dtype(0)), Dtype(nHeight - 1));
				values[s][t] = pdfValue[temp_r*nWidth + temp_c];
				argmax_data[16 * pool_index + s * 4 + t] = temp_r*nWidth + temp_c;
			}
		}
		/*calc the coeff*/
		double u = x - i;
		double v = y - j;
		double A[4], C[4];
		for (int distance = 1, s = 0; distance >= -2; distance--, s++){
			A[s] = cubic_coeff(u + distance);
			C[s] = cubic_coeff(v + distance);
		}
		
		dfCubicValue = 0;
		for (int s = 0; s < 4; s++) {
			for (int t = 0; t < 4; t++) {
				dfCubicValue += values[s][t] * A[t] * C[s];
				w_data[16 * pool_index + s * 4 + t] = A[t] * C[s];
			}
		}
		return dfCubicValue;
	}


	template <typename Dtype>
	void ROIAlignLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		ROIPoolingParameter roi_pool_param = this->layer_param_.roi_pooling_param();
		CHECK_GT(roi_pool_param.pooled_h(), 0)
			<< "pooled_h must be > 0";
		CHECK_GT(roi_pool_param.pooled_w(), 0)
			<< "pooled_w must be > 0";
		pooled_height_ = roi_pool_param.pooled_h();
		pooled_width_ = roi_pool_param.pooled_w();
		spatial_scale_ = roi_pool_param.spatial_scale();
		pad_ratio_ = roi_pool_param.pad_ratio();
		bi_type = roi_pool_param.bi_type();
		is_multi_interpolate = roi_pool_param.is_multi_interpolate();
		LOG(INFO) << "Spatial scale: " << spatial_scale_;

	}

	template <typename Dtype>
	void ROIAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		channels_ = bottom[0]->channels();
		height_ = bottom[0]->height();
		width_ = bottom[0]->width();
		top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
			pooled_width_);
		//[index_lb, index_rb, index_lt, index_rt, w_lb, w_rb, w_lt, w_rt] for each top pixel
		if (bi_type == BiCubic) {
			bili_idx.Reshape(bottom[1]->num(), channels_, pooled_height_,
				pooled_width_ * 16);
			bili_w.Reshape(bottom[1]->num(), channels_, pooled_height_,
				pooled_width_ * 16);
		}
		else {
			bili_idx.Reshape(bottom[1]->num(), channels_, pooled_height_,
				pooled_width_ * 4);
			bili_w.Reshape(bottom[1]->num(), channels_, pooled_height_,
				pooled_width_ * 4);
		}
	}

	template <typename Dtype>
	void ROIAlignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* bottom_rois = bottom[1]->cpu_data();
		// Number of ROIs
		int num_rois = bottom[1]->num();
		int batch_size = bottom[0]->num();
		int top_count = top[0]->count();
		Dtype* top_data = top[0]->mutable_cpu_data();
		caffe_set(top_count, Dtype(-FLT_MAX), top_data);
		int* argmax_data = bili_idx.mutable_cpu_data();
		Dtype* w_data = bili_w.mutable_cpu_data();
		if (bi_type == BiCubic) {
			caffe_set(top_count * 16, -1, argmax_data);
			caffe_set(top_count * 16, Dtype(0), w_data);
		}
		else {
			caffe_set(top_count * 4, -1, argmax_data);
			caffe_set(top_count * 4, Dtype(0), w_data);
		}
		// For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
		for (int n = 0; n < num_rois; ++n) {
			int roi_batch_ind = bottom_rois[0];
			CHECK_GE(roi_batch_ind, 0);
			CHECK_LT(roi_batch_ind, batch_size);

			// padding
			Dtype pad_w, pad_h;
			pad_w = (bottom_rois[3] - bottom_rois[1] + 1)*pad_ratio_;
			pad_h = (bottom_rois[4] - bottom_rois[2] + 1)*pad_ratio_;
			Dtype roi_start_w = (bottom_rois[1] - pad_w) * spatial_scale_;
			Dtype roi_start_h = (bottom_rois[2] - pad_h) * spatial_scale_;
			Dtype roi_end_w = (bottom_rois[3] + pad_w) * spatial_scale_;
			Dtype roi_end_h = (bottom_rois[4] + pad_h) * spatial_scale_;
			// clipping
			roi_start_w = max(roi_start_w, Dtype(0)); roi_start_h = max(roi_start_h, Dtype(0));
			int img_width = round(width_ / spatial_scale_);
			int img_height = round(height_ / spatial_scale_);
			roi_end_w = min(Dtype(img_width - 1), roi_end_w);
			roi_end_h = min(Dtype(img_height - 1), roi_end_h);

			Dtype roi_height = max(roi_end_h - roi_start_h + 1, Dtype(1));
			Dtype roi_width = max(roi_end_w - roi_start_w + 1, Dtype(1));
			const Dtype bin_size_h = static_cast<Dtype>(roi_height)
				/ static_cast<Dtype>(pooled_height_);
			const Dtype bin_size_w = static_cast<Dtype>(roi_width)
				/ static_cast<Dtype>(pooled_width_);

			const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

			if (bi_type == BiCubic) {
				for (int c = 0; c < channels_; ++c) {
					for (int ph = 0; ph < pooled_height_; ++ph) {
						for (int pw = 0; pw < pooled_width_; ++pw) {
							Dtype hcenter = static_cast<Dtype>(ph + 0.5)* bin_size_h;
							Dtype wcenter = static_cast<Dtype>(pw + 0.5)* bin_size_w;
							hcenter = min(max(hcenter + roi_start_h, Dtype(0)), Dtype(height_ - 1));
							wcenter = min(max(wcenter + roi_start_w, Dtype(0)), Dtype(width_ - 1));
							const int pool_index = ph * pooled_width_ + pw;
							top_data[pool_index] = ResampleCubic(wcenter, hcenter, batch_data, width_, height_, pool_index, argmax_data, w_data);
						}
					}
					// Increment all data pointers by one channel
					batch_data += bottom[0]->offset(0, 1);
					top_data += top[0]->offset(0, 1);
					argmax_data += bili_idx.offset(0, 1);
					w_data += bili_w.offset(0, 1);
				}
			}
			else {
				Dtype fX0;
				Dtype fX1;
				Dtype fY0;
				Dtype fY1;
				Dtype fFactorA;
				Dtype fFactorB;
				Dtype fFactorC;
				Dtype fFactorD;

				for (int c = 0; c < channels_; ++c) {
					for (int ph = 0; ph < pooled_height_; ++ph) {
						for (int pw = 0; pw < pooled_width_; ++pw) {
							// Compute pooling region for this output unit:
							//  start (included) = floor(ph * roi_height / pooled_height_)
							//  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
							Dtype hcenter = static_cast<Dtype>(ph + 0.5)* bin_size_h;
							Dtype wcenter = static_cast<Dtype>(pw + 0.5)* bin_size_w;

							hcenter = min(max(hcenter + roi_start_h, Dtype(0)), Dtype(height_ - 1));
							wcenter = min(max(wcenter + roi_start_w, Dtype(0)), Dtype(width_ - 1));

							int hstart = min(max(hcenter, Dtype(0)), Dtype(height_ - 1));
							int wstart = min(max(wcenter, Dtype(0)), Dtype(width_ - 1));
							int hend = min(max(hstart + 1, 0), height_ - 1);
							int wend = min(max(wstart + 1, 0), width_ - 1);

							const int pool_index = ph * pooled_width_ + pw;

							fX0 = wcenter - wstart;
							fX1 = wend - wcenter;
							fY0 = hcenter - hstart;
							fY1 = hend - hcenter;
							fFactorA = fY1 * fX1;
							fFactorB = fY1 * fX0;
							fFactorC = fY0 * fX1;
							fFactorD = fY0 * fX0;

							top_data[pool_index] = batch_data[hstart * width_ + wstart] * fFactorA
								+ batch_data[hstart * width_ + wend] * fFactorB
								+ batch_data[hend * width_ + wstart] * fFactorC
								+ batch_data[hend * width_ + wend] * fFactorD;
							//[index_lb, index_rb, index_lt, index_rt, , w_lb, w_rb, w_lt, w_rt] for each top pixel
							argmax_data[4 * pool_index + 0] = hstart * width_ + wstart;
							argmax_data[4 * pool_index + 1] = hstart * width_ + wend;
							argmax_data[4 * pool_index + 2] = hend * width_ + wstart;
							argmax_data[4 * pool_index + 3] = hend * width_ + wend;
							w_data[4 * pool_index + 0] = fFactorA;
							w_data[4 * pool_index + 1] = fFactorB;
							w_data[4 * pool_index + 2] = fFactorC;
							w_data[4 * pool_index + 3] = fFactorD;
						}
					}
					// Increment all data pointers by one channel
					batch_data += bottom[0]->offset(0, 1);
					top_data += top[0]->offset(0, 1);
					argmax_data += bili_idx.offset(0, 1);
					w_data += bili_w.offset(0, 1);
				}
			}
			// Increment ROI data pointer
			bottom_rois += bottom[1]->offset(1);
		}
	}

	template <typename Dtype>
	void ROIAlignLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[1]) {
			LOG(FATAL) << this->type()
				<< " Layer cannot backpropagate to roi inputs.";
		}
		if (!propagate_down[0]) {
			return;
		}
		const Dtype* bottom_rois = bottom[1]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		caffe_set(bottom[0]->count(), Dtype(0.), bottom_diff);
		const int* argmax_data = bili_idx.cpu_data();
		const int num_rois = top[0]->num();
		const Dtype* w_data = bili_w.cpu_data();
		int argmax_index[16];
		Dtype w[16];
		int w_num = 4;
		if (bi_type == BiCubic) {
			w_num = 16;
		}

		// Accumulate gradient over all ROIs
		for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
			int roi_batch_ind = bottom_rois[roi_n * 5];
			// Accumulate gradients over each bin in this ROI
			for (int c = 0; c < channels_; ++c) {
				for (int ph = 0; ph < pooled_height_; ++ph) {
					for (int pw = 0; pw < pooled_width_; ++pw) {
						int offset_top = ((roi_n * channels_ + c) * pooled_height_ + ph)
							* pooled_width_ + pw;
						for (int index = 0; index < w_num; ++index) {
							argmax_index[index] = argmax_data[offset_top * w_num + index];
							w[index] = w_data[offset_top * w_num + index];
						}
						for (int index = 0; index < w_num; ++index) {
							if (argmax_index[index] >= 0) {
								int offset_bottom = (roi_batch_ind * channels_ + c) * height_
									* width_ + argmax_index[index];
								bottom_diff[offset_bottom] += top_diff[offset_top] * w[index];
							}
						}
					}
				}
			}
		}
	}


#ifdef CPU_ONLY
	STUB_GPU(ROIAlignLayer);
#endif

	INSTANTIATE_CLASS(ROIAlignLayer);
	REGISTER_LAYER_CLASS(ROIAlign);

}  // namespace caffe
