#include <algorithm>
#include <iostream>

#include "caffe/common.hpp"
#include "deformable_im2col.hpp"
using namespace std;

namespace caffe {

template <typename Dtype>
Dtype deformable_im2col_bilinear(const Dtype* bottom_data, const int data_width, 
  const int height, const int width, Dtype h, Dtype w) {

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high;
  int w_high;
  if (h_low >= height - 1) {
	h_high = h_low = height - 1;
	h = (Dtype)h_low;
  }
  else {
	h_high = h_low + 1;
  }

  if (w_low >= width - 1) {
	w_high = w_low = width - 1;
	w = (Dtype)w_low;
  }
  else {
	w_high = w_low + 1;
  }

  Dtype lh = h - h_low;
  Dtype lw = w - w_low;
  Dtype hh = 1 - lh, hw = 1 - lw;

  Dtype v1 = bottom_data[h_low * data_width + w_low];
  Dtype v2 = bottom_data[h_low * data_width + w_high];
  Dtype v3 = bottom_data[h_high * data_width + w_low];
  Dtype v4 = bottom_data[h_high * data_width + w_high];
  Dtype w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  Dtype val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}


template <typename Dtype>
Dtype get_gradient_weight(Dtype argmax_h, Dtype argmax_w, 
  const int h, const int w, const int height, const int width) {

  if (argmax_h < 0 || argmax_h > height || argmax_w < 0 || argmax_w > width) {
	//empty
	return 0;
  }

  argmax_h = max(argmax_h, (Dtype)0.0f);
  argmax_w = max(argmax_w, (Dtype)0.0f);

  int argmax_h_low = (int)argmax_h;
  int argmax_w_low = (int)argmax_w;
  int argmax_h_high;
  int argmax_w_high;
  if (argmax_h_low >= height - 1) {
	argmax_h_high = argmax_h_low = height - 1;
	argmax_h = (Dtype)argmax_h_low;
  } else {
	argmax_h_high = argmax_h_low + 1;
  }
  if (argmax_w_low >= width - 1)
  {
	argmax_w_high = argmax_w_low = width - 1;
	argmax_w = (Dtype)argmax_w_low;
  } else {
	argmax_w_high = argmax_w_low + 1;
  }
  Dtype weight = 0;
  if (h == argmax_h_low) {
	if (w == argmax_w_low) {
	  weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
	} else if (w == argmax_w_high) {
	  weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
	}
  } else if (h == argmax_h_high) {
	if (w == argmax_w_low) {
	  weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
	} else if (w == argmax_w_high) {
	  weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
	}
  }
  return weight;
}


template <typename Dtype>
Dtype get_coordinate_weight(Dtype argmax_h, Dtype argmax_w,
  const int height, const int width, const Dtype* im_data,
  const int data_width, const int bp_dir) {

  if (argmax_h < 0 || argmax_h > height || argmax_w < 0 || argmax_w > width)
  {
	//empty
	return 0;
  }

  if (argmax_h < 0) argmax_h = 0;
  if (argmax_w < 0) argmax_w = 0;

  int argmax_h_low = (int)argmax_h;
  int argmax_w_low = (int)argmax_w;
  int argmax_h_high;
  int argmax_w_high;
  if (argmax_h_low >= height - 1) {
	argmax_h_high = argmax_h_low = height - 1;
	argmax_h = (Dtype)argmax_h_low;
  } else {
	argmax_h_high = argmax_h_low + 1;
  }
  if (argmax_w_low >= width - 1) {
	argmax_w_high = argmax_w_low = width - 1;
	argmax_w = (Dtype)argmax_w_low;
  } else {
	argmax_w_high = argmax_w_low + 1;
  }
  Dtype weight = 0;

  if (bp_dir == 0) {
	weight += -1 * (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_low * data_width + argmax_w_low];
	weight += -1 * (argmax_w - argmax_w_low) * im_data[argmax_h_low * data_width + argmax_w_high];
	weight += (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_high * data_width + argmax_w_low];
	weight += (argmax_w - argmax_w_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
	weight += -1 * (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_low];
	weight += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_high];
	weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_low];
	weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}


/*!
 * \brief deformable_im2col kernel.
 * DO NOT call this directly. Use wrapper function im2col() instead;
 */
template <typename Dtype>
void deformable_im2col_cpu_kernel(const int n, const Dtype* data_im, const Dtype* data_offset,
  const int height, const int width, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  const int channel_per_deformable_group,
  const int height_col, const int width_col,
  Dtype* data_col) {
  //CUDA_KERNEL_LOOP(index, n) {
  #pragma omp parallel for
  for (int index = 0; index < n; index++) {
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int c_im = (index / width_col) / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    
    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;
    
    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    Dtype* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_in) * width + w_in;//0
    
    const Dtype* data_offset_ptr = data_offset;
    data_offset_ptr += deformable_group_index * 2 * kernel_h * kernel_w * height_col * width_col;//0
    
    
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
    	const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
    	const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
    	const Dtype offset_h = data_offset_ptr[data_offset_h_ptr];
    	const Dtype offset_w = data_offset_ptr[data_offset_w_ptr];
    	Dtype val = static_cast<Dtype>(0);
    	const Dtype h_im = h_in + i * dilation_h + offset_h;
    	const Dtype w_im = w_in + j * dilation_w + offset_w;
    	if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
    	  const Dtype map_h = i * dilation_h + offset_h;
    	  const Dtype map_w = j * dilation_w + offset_w;
    	  const int cur_height = height - h_in;
    	  const int cur_width = width - w_in;
    	  val = deformable_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
    	}
    	*data_col_ptr = val;
    	data_col_ptr += height_col * width_col;
      }
    }
  }
}
template <typename Dtype>
void deformable_im2col_cpu(const Dtype* data_im, const Dtype* data_offset, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const int deformable_group,
	Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h -
	  (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
	  (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  int channel_per_deformable_group =  channels/ deformable_group;
  deformable_im2col_cpu_kernel<Dtype>(
	  num_kernels, data_im, data_offset, height, width, kernel_h, kernel_w, pad_h,
	  pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group, height_col,
    width_col, data_col);

}

template void deformable_im2col_cpu<float>(const float* data_im, const float* data_offset, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const int deformable_group,
	float* data_col);
template void deformable_im2col_cpu<double>(const double* data_im, const double* data_offset, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const int deformable_group,
	double* data_col);

}
