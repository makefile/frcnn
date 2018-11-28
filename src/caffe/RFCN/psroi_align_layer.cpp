// ------------------------------------------------------------------
// R-FCN
// Written by Jiangfan Deng
// Written by Afanti<afanti.deng@gmail.com>
// Modify by fyk to add CPU forward
// ------------------------------------------------------------------

#include <algorithm>
#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "psroi_align_layer.hpp"

using std::max;
using std::min;

namespace caffe {
    template <typename Dtype>
        void PSROIAlignLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) {
            PSROIAlignParameter psroi_align_param =
                this->layer_param_.psroi_align_param();
            spatial_scale_ = psroi_align_param.spatial_scale();
            LOG(INFO) << "Spatial scale: " << spatial_scale_;

            CHECK_GT(psroi_align_param.output_dim(), 0)
                << "output_dim must be > 0";
            CHECK_GT(psroi_align_param.group_size(), 0)
                << "group_size must be > 0";

            output_dim_ = psroi_align_param.output_dim();
            group_size_ = psroi_align_param.group_size();
            sample_num_ = psroi_align_param.sample_num();
            pooled_height_ = group_size_;
            pooled_width_ = group_size_;
        }

    template <typename Dtype>
        void PSROIAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) {
            channels_ = bottom[0]->channels();
            CHECK_EQ(channels_, output_dim_*group_size_*group_size_)
                << "input channel number does not match layer parameters";
            height_ = bottom[0]->height();
            width_ = bottom[0]->width();
            top[0]->Reshape(bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
            mapping_channel_.Reshape(bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
            sample_pos_.Reshape(bottom[1]->num(), output_dim_, pooled_height_*pooled_width_*sample_num_*sample_num_, 2);
        }

    template <typename Dtype>
        void bilinear_interpolate(const Dtype* bottom_data, const int height, const int width, Dtype h, Dtype w, Dtype & val) {

            // deal with cases that inverse elements are out of feature map boundary
            if (h < -0.5 || h > height - 0.5 || w < -0.5 || w > width - 0.5) return;

            if (h <= 0) h = 0;
            if (w <= 0) w = 0;


            int h_high;             // h_high 是比 h 大的最小整数
            int w_high;             // w_high 是比 w 大的最小整数
            int h_low = (int) h;    // h_low  是比 h 小的最大整数
            int w_low = (int) w;    // w_low  是比 w 小的最大整数

            if (w_low >= width - 1) {
                w_low = width - 1;
                w_high = width-1;
                w = (Dtype) w_low;
            } else 
                w_high = w_low + 1;

            if (h_low >= height - 1) {
                h_high = height-1;
                h_low = height - 1;
                h = (Dtype) h_low;
            } else 
                h_high = h_low + 1;


            Dtype l_dh = h - h_low;
            Dtype l_dw = w - w_low;
            Dtype h_dh = 1 - l_dh, h_dw = 1 - l_dw;

            // 进行双线性内插
            Dtype u1 = bottom_data[h_low * width + w_low];
            Dtype u2 = bottom_data[h_low * width + w_high];
            Dtype u3 = bottom_data[h_high * width + w_low];
            Dtype u4 = bottom_data[h_high * width + w_high];
            Dtype w1 = h_dh * h_dw, w2 = h_dh * l_dw, w3 = l_dh * h_dw, w4 = l_dh * l_dw;

            val = (w1 * u1 + w2 * u2 + w3 * u3 + w4 * u4);
        }

    template <typename Dtype>
        void PSROIAlignForward(
                const int num,
                const Dtype* bottom_data,
                const Dtype spatial_scale,
                const int channels,
                const int height, const int width,
                const int pooled_height, const int pooled_width,
                const Dtype* bottom_rois,
                const int output_dim,  // 输出通道数
                const int group_size,  // k*k*(c+1) 中的 k
                Dtype* top_data,
                int* mapping_channel,
                Dtype* sample_pos_data,
                const int sample_num) {
            #pragma omp parallel for
            for (int n = 0; n < num; ++n) {
                // [start, end) interval for spatial sampling
                int roi_add = n*5;
                int roi_batch_ind = bottom_rois[roi_add];
                Dtype roi_start_w =
                    static_cast<Dtype>(bottom_rois[roi_add + 1]) * spatial_scale;
                Dtype roi_start_h =
                    static_cast<Dtype>(bottom_rois[roi_add + 2]) * spatial_scale;
                Dtype roi_end_w =
                    static_cast<Dtype>(bottom_rois[roi_add + 3] + 1.) * spatial_scale;
                Dtype roi_end_h =
                    static_cast<Dtype>(bottom_rois[roi_add + 4] + 1.) * spatial_scale;

                // Force too small ROIs to be 1x1
                Dtype roi_width = max<Dtype>(roi_end_w - roi_start_w, 0.1);  // avoid 0
                Dtype roi_height = max<Dtype>(roi_end_h - roi_start_h, 0.1);

                // Compute w and h at bottom
                Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
                Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

                for (int ctop = 0; ctop < output_dim; ++ctop) {
                    for (int ph = 0; ph < pooled_height; ++ph) {
                        for (int pw = 0; pw < pooled_width; ++pw) {
                            int index = n*output_dim*pooled_height*pooled_width + ctop*pooled_height*pooled_width + ph*pooled_width + pw;
                            // 获得当前RoI的宽和高在池化前特征图上的起始和结束索引值, 浮点数
                            Dtype hstart = static_cast<Dtype>(ph) * bin_size_h;
                            Dtype wstart = static_cast<Dtype>(pw) * bin_size_w;
                            Dtype hend   = static_cast<Dtype>(ph + 1) * bin_size_h;
                            Dtype wend   = static_cast<Dtype>(pw + 1) * bin_size_w;

                            // Add roi offsets and clip to input boundaries
                            hstart = min(max(hstart + roi_start_h, Dtype(0)), Dtype(height-1));
                            hend = min(max(hend + roi_start_h, Dtype(0)), Dtype(height-1));
                            wstart = min(max(wstart + roi_start_w, Dtype(0)), Dtype(width-1));
                            wend = min(max(wend + roi_start_w, Dtype(0)), Dtype(width-1));
                            bool is_empty = (hend <= hstart) || (wend <= wstart);

                            int gw = pw;
                            int gh = ph;
                            int c = (ctop*group_size + gh)*group_size + gw; // 

                            // 在池化前特征图上采样点之间的距离，浮点数 (在 h 和 w 两个方向上)
                            Dtype sample_h = bin_size_h / (sample_num + 1);
                            Dtype sample_w = bin_size_w / (sample_num + 1);
                            Dtype val = 0;
                            //bottom_data += (roi_batch_ind * channels + c) * height * width;
                            int bottom_data_offset = (roi_batch_ind * channels + c) * height * width;
                            Dtype out_sum = 0.0;
                            int p_counter = -1;
                            Dtype* sample_pos_diff = sample_pos_data + index * sample_num * sample_num * 2;
                            for (int i = 1; i <= sample_num; ++i) {
                                for (int j = 1; j <= sample_num; ++j) {
                                    ++p_counter;
                                    Dtype cur_h = hstart + i * sample_h;
                                    Dtype cur_w = wstart + j * sample_w;
                                    if (cur_h >= hend || cur_w >= wend) continue;
                                    bilinear_interpolate(bottom_data + bottom_data_offset, height, width, cur_h, cur_w, val);
                                    out_sum += val; 
                                    sample_pos_diff[p_counter * 2 + 0] = cur_h;
                                    sample_pos_diff[p_counter * 2 + 1] = cur_w; 
                                    // updated = true;
                                }
                            }
                            // Dtype bin_area = (hend - hstart) * (wend - wstart);
                            top_data[index] = is_empty ? 0. : out_sum / static_cast<Dtype>(sample_num * sample_num);
                            mapping_channel[index] = c;
                        }
                    }
                }
            }
        }

    template <typename Dtype>
        void PSROIAlignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) {
            const Dtype* bottom_data = bottom[0]->cpu_data();
            const Dtype* bottom_rois = bottom[1]->cpu_data();
            Dtype* top_data = top[0]->mutable_cpu_data();
            Dtype* sample_pos_data = sample_pos_.mutable_cpu_data();
            int* mapping_channel_ptr = mapping_channel_.mutable_cpu_data();
            int count = top[0]->count();
            caffe_set(count, Dtype(0), top_data);
            caffe_set(count, -1, mapping_channel_ptr);
            // NOLINT_NEXT_LINE(whitespace/operators)
            PSROIAlignForward(bottom[1]->num(), bottom_data, spatial_scale_,
                    channels_, height_, width_, pooled_height_,
                    pooled_width_, bottom_rois, output_dim_, group_size_,
                    top_data, mapping_channel_ptr, sample_pos_data, sample_num_);
        }

    template <typename Dtype>
        void PSROIAlignLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
            NOT_IMPLEMENTED;
        }
#ifdef CPU_ONLY
    STUB_GPU(PSROIAlignLayer);
#endif
    INSTANTIATE_CLASS(PSROIAlignLayer);
    REGISTER_LAYER_CLASS(PSROIAlign);

}  // namespace caffe
