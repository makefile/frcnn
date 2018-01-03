
#include <cassert>
#include "data_utils.hpp"
//#define STB_IMAGE_IMPLEMENTATION
//#include "stb_image.h"

// generate rand float point between [min,max], the param can exchange to max,min
float rand_uniform(float min, float max)
{
	if (max < min){
		float swap = min;
		min = max;
		max = swap;
	}
	return ((float)rand() / RAND_MAX * (max - min)) + min;
}

float rand_scale(float s)
{
	float scale = rand_uniform(1, s);
	if (rand() % 2) return scale;
	return 1. / scale;
}
float constrain(float min, float max, float a)
{
	if (a < min) return min;
	if (a > max) return max;
	return a;
}
static float get_pixel(image m, int x, int y, int c)
{
	assert(x < m.w && y < m.h && c < m.c);
	return m.data[c*m.h*m.w + y*m.w + x];
}
static void set_pixel(image m, int x, int y, int c, float val)
{
	if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
	assert(x < m.w && y < m.h && c < m.c);
	m.data[c*m.h*m.w + y*m.w + x] = val;
}
static void add_pixel(image m, int x, int y, int c, float val)
{
	assert(x < m.w && y < m.h && c < m.c);
	m.data[c*m.h*m.w + y*m.w + x] += val;
}
static float get_pixel_extend(image m, int x, int y, int c)
{
	if (x < 0 || x >= m.w || y < 0 || y >= m.h) return 0;
	if (c < 0 || c >= m.c) return 0;
	return get_pixel(m, x, y, c);
}
void constrain_image(image im)
{
	int i;
	for (i = 0; i < im.w*im.h*im.c; ++i){
		if (im.data[i] < 0) im.data[i] = 0;
		if (im.data[i] > 1) im.data[i] = 1;
	}
}
static float bilinear_interpolate(image im, float x, float y, int c)
{
	int ix = (int)floorf(x);
	int iy = (int)floorf(y);

	float dx = x - ix;
	float dy = y - iy;

	float val = (1 - dy) * (1 - dx) * get_pixel_extend(im, ix, iy, c) +
		dy     * (1 - dx) * get_pixel_extend(im, ix, iy + 1, c) +
		(1 - dy) *   dx   * get_pixel_extend(im, ix + 1, iy, c) +
		dy     *   dx   * get_pixel_extend(im, ix + 1, iy + 1, c);
	return val;
}
void fill_image(image m, float s)
{
	int i;
	for (i = 0; i < m.h*m.w*m.c; ++i) m.data[i] = s;
}
float three_way_max(float a, float b, float c)
{
	return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c);
}

float three_way_min(float a, float b, float c)
{
	return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
}
//crop scaled image's top-left area
void place_image(image im, int w, int h, int dx, int dy, image canvas)
{
	int x, y, c;
	for (c = 0; c < im.c; ++c){
		for (y = 0; y < h; ++y){
			for (x = 0; x < w; ++x){
				int rx = ((float)x / w) * im.w;
				int ry = ((float)y / h) * im.h;
				float val = bilinear_interpolate(im, rx, ry, c);
				set_pixel(canvas, x + dx, y + dy, c, val);
			}
		}
	}
}
// http://www.cs.rit.edu/~ncs/color/t_convert.html
void rgb_to_hsv(image im)
{
	assert(im.c == 3);
	int i, j;
	float r, g, b;
	float h, s, v;
	for (j = 0; j < im.h; ++j){
		for (i = 0; i < im.w; ++i){
			r = get_pixel(im, i, j, 0);
			g = get_pixel(im, i, j, 1);
			b = get_pixel(im, i, j, 2);
			float max = three_way_max(r, g, b);
			float min = three_way_min(r, g, b);
			float delta = max - min;
			v = max;
			if (max == 0){
				s = 0;
				h = 0;
			}
			else{
				s = delta / max;
				if (r == max){
					h = (g - b) / delta;
				}
				else if (g == max) {
					h = 2 + (b - r) / delta;
				}
				else {
					h = 4 + (r - g) / delta;
				}
				if (h < 0) h += 6;
				h = h / 6.;
			}
			set_pixel(im, i, j, 0, h);
			set_pixel(im, i, j, 1, s);
			set_pixel(im, i, j, 2, v);
		}
	}
}

void hsv_to_rgb(image im)
{
	assert(im.c == 3);
	int i, j;
	float r, g, b;
	float h, s, v;
	float f, p, q, t;
	for (j = 0; j < im.h; ++j){
		for (i = 0; i < im.w; ++i){
			h = 6 * get_pixel(im, i, j, 0);
			s = get_pixel(im, i, j, 1);
			v = get_pixel(im, i, j, 2);
			if (s == 0) {
				r = g = b = v;
			}
			else {
				int index = floor(h);
				f = h - index;
				p = v*(1 - s);
				q = v*(1 - s*f);
				t = v*(1 - s*(1 - f));
				if (index == 0){
					r = v; g = t; b = p;
				}
				else if (index == 1){
					r = q; g = v; b = p;
				}
				else if (index == 2){
					r = p; g = v; b = t;
				}
				else if (index == 3){
					r = p; g = q; b = v;
				}
				else if (index == 4){
					r = t; g = p; b = v;
				}
				else {
					r = v; g = p; b = q;
				}
			}
			set_pixel(im, i, j, 0, r);
			set_pixel(im, i, j, 1, g);
			set_pixel(im, i, j, 2, b);
		}
	}
}
void scale_image_channel(image im, int c, float v)
{
	int i, j;
	for (j = 0; j < im.h; ++j){
		for (i = 0; i < im.w; ++i){
			float pix = get_pixel(im, i, j, c);
			pix = pix*v;
			set_pixel(im, i, j, c, pix);
		}
	}
}
void distort_image(image im, float hue, float sat, float val)
{
	rgb_to_hsv(im);
	scale_image_channel(im, 1, sat);
	scale_image_channel(im, 2, val);
	int i;
	for (i = 0; i < im.w*im.h; ++i){
		im.data[i] = im.data[i] + hue;
		if (im.data[i] > 1) im.data[i] -= 1;
		if (im.data[i] < 0) im.data[i] += 1;
	}
	hsv_to_rgb(im);
	constrain_image(im);
}
void random_distort_image(image im, float hue, float saturation, float exposure)
{
	float dhue = rand_uniform(-hue, hue);
	float dsat = rand_scale(saturation);
	float dexp = rand_scale(exposure);
	distort_image(im, dhue, dsat, dexp);
}
void flip_image(image a)
{
	int i, j, k;
	for (k = 0; k < a.c; ++k){
		for (i = 0; i < a.h; ++i){
			for (j = 0; j < a.w / 2; ++j){
				int index = j + a.w*(i + a.h*(k));
				int flip = (a.w - j - 1) + a.w*(i + a.h*(k));
				float swap = a.data[flip];
				a.data[flip] = a.data[index];
				a.data[index] = swap;
			}
		}
	}
}
image make_empty_image(int w, int h, int c)
{
	image out;
	out.data = 0;
	out.h = h;
	out.w = w;
	out.c = c;
	return out;
}

image make_image(int w, int h, int c)
{
	image out = make_empty_image(w, h, c);
	out.data = (float*)calloc(h*w*c, sizeof(float));
	return out;
}
void free_image(image m)
{
	if (m.data){
		free(m.data);
	}
}
image resize_image(image im, int w, int h)
{
	image resized = make_image(w, h, im.c);
	image part = make_image(w, im.h, im.c);
	int r, c, k;
	float w_scale = (float)(im.w - 1) / (w - 1);
	float h_scale = (float)(im.h - 1) / (h - 1);
	for (k = 0; k < im.c; ++k){
		for (r = 0; r < im.h; ++r){
			for (c = 0; c < w; ++c){
				float val = 0;
				if (c == w - 1 || im.w == 1){
					val = get_pixel(im, im.w - 1, r, k);
				}
				else {
					float sx = c*w_scale;
					int ix = (int)sx;
					float dx = sx - ix;
					val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix + 1, r, k);
				}
				set_pixel(part, c, r, k, val);
			}
		}
	}
	for (k = 0; k < im.c; ++k){
		for (r = 0; r < h; ++r){
			float sy = r*h_scale;
			int iy = (int)sy;
			float dy = sy - iy;
			for (c = 0; c < w; ++c){
				float val = (1 - dy) * get_pixel(part, c, iy, k);
				set_pixel(resized, c, r, k, val);
			}
			if (r == h - 1 || im.h == 1) continue;
			for (c = 0; c < w; ++c){
				float val = dy * get_pixel(part, c, iy + 1, k);
				add_pixel(resized, c, r, k, val);
			}
		}
	}

	free_image(part);
	return resized;
}

//transform the boxes label as we do to image data. 
//notice that the boxes label is normalized before this input.
//the bad boxes will be annotated as w,h=-1
void correct_boxes(box_label *boxes, int n, float dx, float dy, float sx, float sy, int flip)
{
	int i;
	for (i = 0; i < n; ++i){
		if (boxes[i].x == 0 && boxes[i].y == 0) { // fyk: this will not affect the network learning
			boxes[i].w = -1;
			boxes[i].h = -1;
			std::cout << "=======WARNING====correct_boxes=====this shouldn't happen!" << std::endl;
			continue;
		}
		boxes[i].left = boxes[i].left  * sx - dx;
		boxes[i].right = boxes[i].right * sx - dx;
		boxes[i].top = boxes[i].top   * sy - dy;
		boxes[i].bottom = boxes[i].bottom* sy - dy;

		if (flip){
			float swap = boxes[i].left;
			boxes[i].left = 1. - boxes[i].right;
			boxes[i].right = 1. - swap;
		}

		boxes[i].left = constrain(0, 1, boxes[i].left);
		boxes[i].right = constrain(0, 1, boxes[i].right);
		boxes[i].top = constrain(0, 1, boxes[i].top);
		boxes[i].bottom = constrain(0, 1, boxes[i].bottom);

		boxes[i].x = (boxes[i].left + boxes[i].right) / 2;
		boxes[i].y = (boxes[i].top + boxes[i].bottom) / 2;
		boxes[i].w = (boxes[i].right - boxes[i].left);
		boxes[i].h = (boxes[i].bottom - boxes[i].top);

		boxes[i].w = constrain(0, 1, boxes[i].w);
		boxes[i].h = constrain(0, 1, boxes[i].h);
		//		std::cout << "correct_boxes" << boxes[i].x << ' ' << boxes[i].y << ' ' << boxes[i].w << ' ' << boxes[i].h << std::endl;
		if (boxes[i].left >= boxes[i].right || boxes[i].top >= boxes[i].bottom)
		{
			// fyk: some border situation maybe cannot be handled properly
			boxes[i].w = -1;
			boxes[i].h = -1;
			continue;
		}
	}
}

void rgbgr_image(image im);
//void show_image_cv(image p, const char *name, IplImage *disp);
void ipl_into_image(IplImage* src, image im)
{
	unsigned char *data = (unsigned char *)src->imageData;
	int h = src->height;
	int w = src->width;
	int c = src->nChannels;
	int step = src->widthStep;
	int i, j, k;

	for (i = 0; i < h; ++i){
		for (k = 0; k < c; ++k){
			for (j = 0; j < w; ++j){
				im.data[k*w*h + i*w + j] = data[i*step + j*c + k] / 255.;
			}
		}
	}
	rgbgr_image(im);//convert from BGR to RGB
}

image ipl_to_image(IplImage* src)
{
	int h = src->height;
	int w = src->width;
	int c = src->nChannels;
	image out = make_image(w, h, c);
	ipl_into_image(src, out);
	return out;
}

image cvmat_to_image(cv::Mat &mat) {
	/**
	NOTICE that IplImage's widthStep is different from cv::Mat,see http://answers.opencv.org/question/20407/matstep-iplimagewidthstep/
	so convert from cv::Mat to IplImage then use widthStep will get wrong result
	IplImage ix = IplImage(mat_img);
	IplImage* img = &ix; //share data,only copy header
	cvSaveImage("augment-iplimage.jpg", img);
	//IplImage* img = cvCloneImage(&ix);
	image ret = ipl_to_image(img);
	//cvReleaseImage(&img);
	**/
	cv::Mat mat_img = mat;
	if (mat.type() != CV_32FC3)
	{
		mat.convertTo(mat_img, CV_32FC3);
	}
	int h = mat_img.rows;
	int w = mat_img.cols;
	int c = mat_img.channels();
	float *data = (float *)mat_img.data;
	image im = make_image(w, h, c);
	int i, j, k;
	for (i = 0; i < h; ++i){
		for (j = 0; j < w; ++j){
			for (k = 0; k < c; ++k){
				int cv_offset = (i * w + j) * c;
				im.data[k*w*h + i*w + j] = data[cv_offset + k] / 255.;
			}
		}
	}
	rgbgr_image(im);//convert from BGR to RGB

	return im;
}

image copy_image(image p)
{
	image copy = p;
	copy.data = (float*)calloc(p.h*p.w*p.c, sizeof(float));
	memcpy(copy.data, p.data, p.h*p.w*p.c*sizeof(float));
	return copy;
}
// RGB <---> BGR
void rgbgr_image(image im)
{
	int i;
	for (i = 0; i < im.w*im.h; ++i){
		float swap = im.data[i];
		im.data[i] = im.data[i + im.w*im.h * 2];
		im.data[i + im.w*im.h * 2] = swap;
	}
}

image load_image_cv(const char *filename, int channels)
{
	IplImage* src = 0;
	int flag = -1;
	if (channels == 0) flag = -1;
	else if (channels == 1) flag = 0;
	else if (channels == 3) flag = 1;
	else {
		fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
	}

	if ((src = cvLoadImage(filename, flag)) == 0)
	{
		fprintf(stderr, "Cannot load image \"%s\"\n", filename);
		//		char buff[256];
		//		sprintf_s(buff, "echo %s >> bad.list", filename);
		//		system(buff);
		return make_image(10, 10, 3);
		//exit(0);
	}
	image out = ipl_to_image(src);
	cvReleaseImage(&src);
	return out;
}

image load_image(const char *filename, int w, int h, int c)
{
	//#ifdef OPENCV
	image out = load_image_cv(filename, c);
	//#else
	//	image out = load_image_stb(filename, c);
	//#endif

	if ((h && w) && (h != out.h || w != out.w)){
		image resized = resize_image(out, w, h);
		free_image(out);
		out = resized;
	}
	return out;
}

image load_image_color(const char *filename, int w, int h)
{
	return load_image(filename, w, h, 3);
}

void show_image_cv(image p, const char *name, IplImage *disp)
{
	int x, y, k;
	if (p.c == 3) rgbgr_image(p);
	//normalize_image(copy);

	char buff[256];
	//sprintf(buff, "%s (%d)", name, windows);
	sprintf(buff, "%s", name);

	int step = disp->widthStep;
	//	cvNamedWindow(buff, CV_WINDOW_NORMAL);
	cvNamedWindow(buff, CV_WINDOW_AUTOSIZE);
	//cvMoveWindow(buff, 100*(windows%10) + 200*(windows/10), 100*(windows%10));
	//++windows;
	for (y = 0; y < p.h; ++y){
		for (x = 0; x < p.w; ++x){
			for (k = 0; k < p.c; ++k){
				disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(p, x, y, k) * 255);
			}
		}
	}
	/*if (0){
	int w = 448;
	int h = w*p.h / p.w;
	if (h > 1000){
	h = 1000;
	w = h*p.w / p.h;
	}
	IplImage *buffer = disp;
	disp = cvCreateImage(cvSize(w, h), buffer->depth, buffer->nChannels);
	cvResize(buffer, disp, CV_INTER_LINEAR);
	cvReleaseImage(&buffer);
	}
	*/
	cvShowImage(buff, disp);
}

void show_image(image p, const char *name)
{
	IplImage *disp = cvCreateImage(cvSize(p.w, p.h), IPL_DEPTH_8U, p.c);
	image copy = copy_image(p);
	constrain_image(copy);
	show_image_cv(copy, name, disp);
	free_image(copy);
	cvReleaseImage(&disp);
	//	fprintf(stderr, "Not compiled with OpenCV, saving to %s.png instead\n", name);
	//	save_image(p, name);
}
void save_image_jpg(image p, const char *name)
{
	image copy = copy_image(p);
	if (p.c == 3) rgbgr_image(copy);
	int x, y, k;

	char buff[256];
	sprintf(buff, "%s.jpg", name);

	IplImage *disp = cvCreateImage(cvSize(p.w, p.h), IPL_DEPTH_8U, p.c);
	int step = disp->widthStep;
	for (y = 0; y < p.h; ++y){
		for (x = 0; x < p.w; ++x){
			for (k = 0; k < p.c; ++k){
				disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(copy, x, y, k) * 255);
			}
		}
	}
	cvSaveImage(buff, disp, 0);
	cvReleaseImage(&disp);
	free_image(copy);
}

cv::Mat image2cvmat(image p)
{
	image copy = copy_image(p);
	if (p.c == 3) rgbgr_image(copy);
	int x, y, k;
	/*
	IplImage *disp = cvCreateImage(cvSize(p.w, p.h), IPL_DEPTH_8U, p.c);
	int step = disp->widthStep;
	for (y = 0; y < p.h; ++y){
	for (x = 0; x < p.w; ++x){
	for (k = 0; k < p.c; ++k){
	disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(copy, x, y, k) * 255);
	}
	}
	}
	cv::Mat mat = cv::cvarrToMat(disp, true);//copyData=true
	cvReleaseImage(&disp);
	free_image(copy);
	*/
	//NOTICE that use cv::cvarrToMat and the mat maybe type of 8UC3,and we must convert it to 32FC3,if we want to use it as float value.
	cv::Mat mat = cv::Mat::zeros(cv::Size(p.h, p.w), CV_32FC3);
	float *data = (float *)mat.data;
	for (y = 0; y < p.h; ++y)
		for (x = 0; x < p.w; ++x)
			for (k = 0; k < p.c; ++k)
				data[(y * p.w + x) * p.c + k] = get_pixel(copy, x, y, k) * 255;
	return mat;
}
image rotate_image(image im, float rad)
{
	int x, y, c;
	float cx = im.w / 2.;
	float cy = im.h / 2.;
	image rot = make_image(im.w, im.h, im.c);
	for (c = 0; c < im.c; ++c){
		for (y = 0; y < im.h; ++y){
			for (x = 0; x < im.w; ++x){
				float rx = cos(rad)*(x - cx) - sin(rad)*(y - cy) + cx;
				float ry = sin(rad)*(x - cx) + cos(rad)*(y - cy) + cy;
				if (rx < 0 || ry < 0 || rx >= im.w || ry >= im.h)
				{
					set_pixel(rot, x, y, c, 0.5); // fill_image(rot, 0.5);
				}
				else
				{
					float val = bilinear_interpolate(im, rx, ry, c);
					set_pixel(rot, x, y, c, val);
				}

			}
		}
	}
	return rot;
}

void set_rand_seed(int seed)
{
	if (seed<0)
	{
		seed = time(NULL);
		srand(seed);
	}
	else
		srand(seed);
}
image data_augment(image orig, box_label *boxes, int num_boxes, int w, int h, int flip, float jitter, float hue, float saturation, float exposure) {

	if (w <= 0 || h <= 0)
	{
		w = orig.w;
		h = orig.h;
	}

	image sized = make_image(w, h, orig.c);// output image resized.
	fill_image(sized, .5);// blank image normalized to 0~1,fill with mean value: 0.5

	float dw = jitter * orig.w;
	float dh = jitter * orig.h;

	float new_ar = (orig.w + rand_uniform(-dw, dw)) / (orig.h + rand_uniform(-dh, dh));
	float scale = rand_uniform(.5, 2);// .25 maybe is too small

	float nw, nh;

	// scale according to new aspect ratio
	if (new_ar < 1){
		nh = scale * h;
		nw = nh * new_ar;
	}
	else {
		nw = scale * w;
		nh = nw / new_ar;
	}

	float dx = rand_uniform(0, w - nw);
	float dy = rand_uniform(0, h - nh);

	// jitter: scale and fill data to sized image
	place_image(orig, nw, nh, dx, dy, sized);
	//	free_image(orig);
	// three basic augmentation method
	random_distort_image(sized, hue, saturation, exposure);
	if (flip) flip_image(sized);
	// handle the box lebels
	correct_boxes(boxes, num_boxes, -dx / w, -dy / h, nw / w, nh / h, flip);

	return sized;
}
// return CV_32FC3 image
cv::Mat data_augment(cv::Mat &src, std::vector<std::vector<float> > &rois,
	int flip, float jitter, float hue, float saturation, float exposure) {
	int num_boxes = rois.size();
	box_label *boxes = (box_label*)calloc(num_boxes, sizeof(box_label));
	convert_box(rois, boxes, src.cols, src.rows);
	image orig = cvmat_to_image(src);
	image result = data_augment(orig, boxes, num_boxes, 0, 0, flip, jitter, hue, saturation, exposure);
//	rois = convert_box(boxes, num_boxes, src.cols, src.rows);
	rois = convert_box(boxes, num_boxes, result.w, result.h);
	free(boxes);
	free_image(orig);
	return image2cvmat(result);
}
void rotate180(box_label *label_in, box_label *label_out, int num_boxes)
{
	for (int i = 0; i < num_boxes; i++)
	{
		box_label b = label_in[i];
		label_out[i].x = 1. - b.x;
		label_out[i].y = 1. - b.y;
		label_out[i].w = b.w;
		label_out[i].h = b.h;
		label_out[i].left  = label_out[i].x - label_out[i].w / 2;
		label_out[i].right = label_out[i].x + label_out[i].w / 2;
		label_out[i].top = label_out[i].y - label_out[i].h / 2;
		label_out[i].bottom = label_out[i].y + label_out[i].h / 2;

		label_out[i].left = constrain(0, 1, label_out[i].left);
		label_out[i].right = constrain(0, 1, label_out[i].right);
		label_out[i].top = constrain(0, 1, label_out[i].top);
		label_out[i].bottom = constrain(0, 1, label_out[i].bottom);
	}
}
// only support 90,180 etc up-right degrees,range(0~2pi)
image rotate_augment(float rad, image im_in, box_label *label_in, box_label *label_out, int num_boxes)
{
	image rot = rotate_image(im_in, rad);
	if (fabs(rad - M_PI) < 0.001)
	{
		rotate180(label_in, label_out, num_boxes);
		return rot;
	}
	// handle labels
	for (int i = 0; i < num_boxes; i++)
	{
		box_label b = label_in[i];

		float a_cos = cos(rad);
		float a_sin = -sin(rad);// anti clock-wise
		float cx = im_in.w / 2.;//when calc rect after rotate,cannot use relative value.
		float cy = im_in.h / 2.;
		cv::Point2f pts[4];
		for (int i = 0; i < 4; i++)
		{
			float rx = (i < 2 ? b.left : b.right)*im_in.w;
			float ry = (i % 3 == 0 ? b.top : b.bottom)*im_in.h;
			float x = -(ry - cy) * a_sin + (rx - cx) * a_cos + cx;
			float y = (ry - cy) * a_cos + (rx - cx) * a_sin + cy;
			pts[i] = cv::Point2f(x / im_in.w, y / im_in.h);
		}
		float min_x = std::min(pts[0].x, pts[1].x);
		float max_x = std::max(pts[0].x, pts[1].x);
		float min_y = std::min(pts[1].y, pts[2].y);
		float max_y = std::max(pts[1].y, pts[2].y);

		label_out[i].id = label_in[i].id;

		label_out[i].left = min_x;
		label_out[i].right = max_x;
		label_out[i].top = min_y;
		label_out[i].bottom = max_y;

		label_out[i].left = constrain(0, 1, label_out[i].left);
		label_out[i].right = constrain(0, 1, label_out[i].right);
		label_out[i].top = constrain(0, 1, label_out[i].top);
		label_out[i].bottom = constrain(0, 1, label_out[i].bottom);

		label_out[i].x = (label_out[i].left + label_out[i].right) / 2;
		label_out[i].y = (label_out[i].top + label_out[i].bottom) / 2;
		label_out[i].w = (label_out[i].right - label_out[i].left);
		label_out[i].h = (label_out[i].bottom - label_out[i].top);

		label_out[i].w = constrain(0, 1, label_out[i].w);
		label_out[i].h = constrain(0, 1, label_out[i].h);
	}

	return rot;
}
void convert_box(std::vector<std::vector<float> > &rois, box_label *out_boxes, float img_width, float img_height)
{
	int num_boxes = rois.size();
	for (int i = 0; i < num_boxes; i++)
	{
		out_boxes[i].id = rois[i][0];
		out_boxes[i].left = rois[i][1] / img_width;
		out_boxes[i].right = rois[i][3] / img_width;
		out_boxes[i].top = rois[i][2] / img_height;
		out_boxes[i].bottom = rois[i][4] / img_height;
		out_boxes[i].x = (out_boxes[i].left + out_boxes[i].right) / 2;
		out_boxes[i].y = (out_boxes[i].top + out_boxes[i].bottom) / 2;
		out_boxes[i].h = out_boxes[i].bottom - out_boxes[i].top;
		out_boxes[i].w = out_boxes[i].right - out_boxes[i].left;
	}
}
std::vector<std::vector<float> > convert_box(box_label *boxes, int num_boxes, float img_width, float img_height)
{
	std::vector<std::vector<float> > rois;
	for (int i = 0; i < num_boxes; i++)
	{
		if (boxes[i].w <= 0 || boxes[i].h <= 0)
		{
			continue; // filter out the boxes that outside border after image augmentation
		}
		std::vector<float> t(5);
		t[0] = (boxes[i].id);
		t[1] = (boxes[i].left * img_width);
		t[2] = (boxes[i].top * img_height);
		t[3] = (boxes[i].right * img_width);
		t[4] = (boxes[i].bottom * img_height);
		rois.push_back(t);
	}
	return rois;
}