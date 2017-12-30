#ifndef _IMG_UTILS_H
#define _IMG_UTILS_H

#include <cstdlib>
#include <cstdio>
#define _USE_MATH_DEFINES
#include <cmath>
#include <ctime>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/version.hpp"
#if CV_MAJOR_VERSION == 3
#include "opencv2/videoio/videoio_c.h"
#include "opencv2/imgcodecs/imgcodecs_c.h"
#endif

typedef struct {
	int w;
	int h;
	int c;
	float *data;
} image;

typedef struct{
	int id;
	float x, y, w, h;
	float left, right, top, bottom;
} box_label;

// generate rand float point between [min,max], the param can exchange to max,min
float rand_uniform(float min, float max);
float constrain(float min, float max, float a);
//crop scaled image's top-left area
void place_image(image im, int w, int h, int dx, int dy, image canvas);
void random_distort_image(image im, float hue, float saturation, float exposure);
image make_image(int w, int h, int c);
void free_image(image m);
void flip_image(image a);
void fill_image(image m, float s);
image load_image_color(const char *filename, int w, int h);
void show_image_cv(image p, const char *name, IplImage *disp);
void show_image(image p, const char *name);
void save_image_jpg(image p, const char *name);
cv::Mat image2cvmat(image p);
void cvDrawDottedRect(IplImage* img, CvRect rect, CvScalar color, int lengthOfDots=3, int thickness = 1, int lineType = 8);
void cvDrawDottedRect(cv::Mat& mat_img, cv::Point pt1, cv::Point pt2, cv::Scalar color, int lengthOfDots = 3, int thickness = 1, int lineType = 8);

//anti-clockwise rotation
image rotate_image(image im, float rad);
image rotate_augment(float angle, image im_in, box_label *label_in, box_label *label_out, int num_boxes);
void set_rand_seed(int seed);

image data_augment(image orig, box_label *boxes, int num_boxes, int w, int h, int flip, float jitter, float hue, float saturation, float exposure);
cv::Mat data_augment(cv::Mat &orig, std::vector<std::vector<float> > &boxes, int flip, float jitter, float hue, float saturation, float exposure);
void correct_boxes(box_label *boxes, int n, float dx, float dy, float sx, float sy, int flip);
void convert_box(std::vector<std::vector<float> > &rois, box_label *out_boxes, float img_width, float img_height);
std::vector<std::vector<float> > convert_box(box_label *boxes, int num_boxes, float img_width, float img_height);

#endif
