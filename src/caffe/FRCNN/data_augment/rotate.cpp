#include "data_utils.hpp"
/**
 * code from https://github.com/opencv/opencv/blob/0d6518aaa05bc66b5724844938b6920627c5f13c/modules/core/src/copy.cpp
 * and https://stackoverflow.com/questions/15043152/rotate-opencv-matrix-by-90-180-270-degrees
 */
using namespace cv;
/*void rotate(InputArray src, OutputArray dst, int rotateCode);
enum RotateFlags {
	ROTATE_90_CLOCKWISE = 0, //Rotate 90 degrees clockwise
	ROTATE_180 = 1, //Rotate 180 degrees clockwise
	ROTATE_90_COUNTERCLOCKWISE = 2, //Rotate 270 degrees clockwise
};*/
// black padding, and edges will be rotated to oustide of the image's original size.
void rotate(cv::Mat& src, double couter_clockwise_degree, cv::Mat& dst){
	cv::Point2f ptCp(src.cols*0.5, src.rows*0.5);
	cv::Mat M = cv::getRotationMatrix2D(ptCp, couter_clockwise_degree, 1.0);
	cv::warpAffine(src, dst, M, src.size(), cv::INTER_CUBIC); //Nearest is too rough
}
void rotate_quicker(Mat& src, int clockwise_degree, cv::Mat& dst)
{
	//Mat dst;// = src.clone();
	switch (clockwise_degree)
	{
	case 90: // same as -270
		//cv::rotate(src, dst, ROTATE_90_CLOCKWISE);
		transpose(src, dst);
		flip(dst, dst, 1);
		break;
	case 180:
		//rotate(src, dst, ROTATE_180);
		flip(src, dst, -1);
		break;
	case 270:
		//rotate(src, dst, ROTATE_90_COUNTERCLOCKWISE);
		transpose(src, dst);
		flip(dst, dst, 0);
		break;
	default:
		rotate(src, - clockwise_degree, dst);
	}
	
	//return dst;
}
void matrix_mult(float* a, float* b, float* out, int m, int n, int p)
{
	int i, j, k;
	int temp;
	for (i = 0; i < m; i++){
		for (j = 0; j < p; j++){
			temp = 0;
			for (k = 0; k < n; k++){
				temp += a[i*n+k] * b[k*p+j];
			}
			out[i*p+j] = temp;
		}
	}
}
std::vector<std::vector<float> > rotate_rois_90(cv::Mat &src, std::vector<std::vector<float> > &rois_in, int clockwise_degree)
{
	std::vector<std::vector<float> > rois = rois_in;
	float newH, newW;
	switch (clockwise_degree)
	{
	case 90: // same as -270
		newH = src.cols;
		newW = src.rows;
		for (int i = 0; i < rois.size(); i++) {
			float old_x1 = rois[i][1];
			float old_y1 = rois[i][2];
			float old_x2 = rois[i][3];
			float old_y2 = rois[i][4];
			rois[i][1] = newW - old_y1 - 1;
			rois[i][2] = old_x1;
			rois[i][3] = newW - old_y2 - 1;
			rois[i][4] = old_x2;
		}
		break;
	case 180:
		newH = src.rows;
		newW = src.cols;
		for (int i = 0; i < rois.size(); i++) {
			float old_x1 = rois[i][1];
			float old_y1 = rois[i][2];
			float old_x2 = rois[i][3];
			float old_y2 = rois[i][4];
			rois[i][1] = newW - old_x1 - 1;// newW - old_y1 - 1;
			rois[i][2] = newH - old_y1 - 1;
			rois[i][3] = newW - old_x2 - 1;
			rois[i][4] = newH - old_y2 - 1;
		}
		break;
	case 270:
		newH = src.cols;
		newW = src.rows;
		for (int i = 0; i < rois.size(); i++) {
			float old_x1 = rois[i][1];
			float old_y1 = rois[i][2];
			float old_x2 = rois[i][3];
			float old_y2 = rois[i][4];
			rois[i][1] = old_y1;// newW - old_y1 - 1;
			rois[i][2] = newH - old_x1 - 1;
			rois[i][3] = old_y2;
			rois[i][4] = newH - old_x2 - 1;
		}
		break;
	default:
		break;
	}
	return rois;
}
std::vector<std::vector<float> > rotate_rois(cv::Mat &src, std::vector<std::vector<float> > &rois, int clockwise_degree)
{
	if (clockwise_degree % 90 == 0) {
		return rotate_rois_90(src, rois, clockwise_degree);
	}

	float theta = clockwise_degree * M_PI / 180;
	float R[][2] = { cos(theta), -sin(theta), sin(theta), cos(theta) };
	float pivot[2] = { src.rows / (float)2.0, src.cols / (float)2.0 };
	std::vector<std::vector<float> > newboxes;
	for (int i = 0; i < rois.size(); i++)
	{
		float xmin = rois[i][1];
		float ymin = rois[i][2];
		float xmax = rois[i][3];
		float ymax = rois[i][4];
		xmin -= pivot[1];
		xmax -= pivot[1];
		ymin -= pivot[0];
		ymax -= pivot[0];
		float bfull[][4] = { xmin, xmin, xmax, xmax, ymin, ymax, ymin, ymax };
		float c[2][4];
		matrix_mult((float*)R, (float*)bfull, (float*)c, 2, 2, 4);

		for (int j = 0; j < 4; j++)
		{
			c[0][j] += pivot[1];
			c[0][j] = constrain(0, src.cols, c[0][j]);
			c[1][j] += pivot[0];
			c[1][j] = constrain(0, src.rows, c[1][j]);
		}
		bool c0zero = false;
		bool c1zero = false;
		if (c[1][0] == src.rows || c[1][0] == 0)
		{
			if (c[1][1] == c[1][0] && c[1][2] == c[1][0] && c[1][3] == c[1][0])
			{
				c[0][0] = 0; c[0][1] = 0; c[0][2] = 0; c[0][3] = 0;
				c0zero = true;
			}
		}
		if (c[0][0] == src.cols || c[0][0] == 0)
		{
			if (c[0][1] == c[0][0] && c[0][2] == c[0][0] && c[0][3] == c[0][0])
			{
				c[1][0] = 0; c[1][1] = 0; c[1][2] = 0; c[1][3] = 0;
				c1zero = true;
			}
		}
		if (c0zero && c1zero) continue;
		float c0max = c[0][0];
		float c0min = c[0][0];
		float c1max = c[1][0];
		float c1min = c[1][0];
		for (int j = 1; j < 4; j++)
		{
			if (c[0][j] < c0min) c0min = c[0][j];
			else if (c[0][j] > c0max) c0max = c[0][j];
			if (c[1][j] < c1min) c1min = c[1][j];
			else if (c[1][j] > c1max) c1max = c[1][j];
		}
		float newbox[] = { (float)rois[i][0], c0min, c1min, c0max, c1max };
		newboxes.push_back(std::vector<float>(newbox, newbox + sizeof(newbox)/sizeof(float)));
	}
	return newboxes;
}
std::vector<std::vector<float> > rotate_rois(cv::Mat &src, std::vector<std::vector<float> > &rois, 
	int clockwise_degree, cv::Mat& dst)
{
	rotate_quicker(src, clockwise_degree, dst);
	return rotate_rois(src, rois, clockwise_degree);
}
int _main()
{
	std::string img_path = "100001534.bmp";
	cv::Mat origmat = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
	std::vector<float> tmp{
		2, 131, 307, 239, 615,
		2, 626, 33, 1014, 440,
		2, 808, 546, 869, 800 };

	std::vector<std::vector<float> > rois;
	for (int i = 0; i < tmp.size(); i += 5)
	{
		std::vector<float> tmp1 = { tmp[i], tmp[i + 1], tmp[i + 2], tmp[i + 3], tmp[i + 4] };
		rois.push_back(tmp1);
	}
	for (int i = 0; i < rois.size(); i++)
	{
		cvDrawDottedRect(origmat, cv::Point(rois[i][1], rois[i][2]), cv::Point(rois[i][3], rois[i][4]), cv::Scalar(200, 0, 0), 6, 1);
	}
	vis_32f_mat("orig", origmat);
	//cvWaitKey(0);
//	int clockwise_degree = 70;
	
	for (int clockwise_degree = 90; clockwise_degree < 360; clockwise_degree+=90)
	{
		cv::Mat rotated;
		std::vector<std::vector<float> > aug_rois = rotate_rois(origmat, rois, clockwise_degree, rotated);
		for (int i = 0; i < aug_rois.size(); i++)
		{
			cvDrawDottedRect(rotated, cv::Point(aug_rois[i][1], aug_rois[i][2]), cv::Point(aug_rois[i][3], aug_rois[i][4]), cv::Scalar(0, 0, 200), 6, 1);
		}
		vis_32f_mat("rotated", rotated);
		cvWaitKey(0);
	}
	
	return 0;
}
/**
 * Shift an image by a random amount on the x and y axis 
 * drawn from discrete uniform distribution with parameter
 */
std::vector<std::vector<float> > shift_image(cv::Mat &srcMat, std::vector<std::vector<float> > &rois,
	float jitter, cv::Mat& dst)
{
	if (jitter <= 0.05)
	{
		dst = srcMat.clone();
		return rois;
	}
	cv::Mat src;
	if (srcMat.type() != CV_32FC3)
	{
		srcMat.convertTo(src, CV_32FC3);
	}
	else src = srcMat;

	int w = src.cols;
	int h = src.rows;
	float dw = jitter * w;
	float dh = jitter * h;
	int dx = (int)rand_uniform(-dw, dw);
	int dy = (int)rand_uniform(-dh, dh);
	dst = cv::Mat::zeros(src.size(), src.type());
	std::vector<std::vector<float> > nb;
	for (int i = 0; i < rois.size(); i++) {

		float xmin = constrain(0, w, rois[i][1] + dx);
		float xmax = constrain(0, w, rois[i][3] + dx);
		float ymin = constrain(0, h, rois[i][2] + dy);
		float ymax = constrain(0, h, rois[i][4] + dy);
		//we only add the box if they are not all 0
		if (!(xmin == 0 && xmax == 0 && ymin == 0 && ymax == 0))
		{
			float newbox[] = { (float)rois[i][0], xmin, ymin, xmax, ymax };
			nb.push_back(std::vector<float>(newbox, newbox + sizeof(newbox) / sizeof(float)));
		}
	}
	float *res_mat_data = (float *)src.data;
	float *new_mat_data = (float *)dst.data;
	for (int y = max(dy, 0); y < min(h, h + dy); ++y)
		for (int x = max(dx, 0); x < min(w, w + dx); ++x)
			for (int k = 0; k < src.channels(); ++k)
			{
				int oy = y - dy;
				int ox = x - dx;
				new_mat_data[(y * w + x) * src.channels() + k] = 
					res_mat_data[(oy * w + ox) * src.channels() + k];
			}
	
	return nb;
}
