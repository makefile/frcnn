#include "iostream"
#include <algorithm>
#include "time.h"
#include "string.h"
//#include "io.h"

/****** OpenCV *******/
#include "opencv2/opencv.hpp"

#define MAX_INT 20000000

//Type of Min and Max value
typedef struct _MinMax
{
	double min;
	double max;
}MinMax;

cv::Mat ReadImage();
void rerange();
void fill_x_y();
int find_table(int y);
void locate(int l1, int l2, double l3);
void getL(cv::Mat img);

cv::Vec<float, 3> Airlight(cv::Mat img, cv::Mat dark);
cv::Mat TransmissionMat(cv::Mat dark);
cv::Mat DarkChannelPrior(cv::Mat img);

void RefineTrans(cv::Mat trans);


void printMat(char * name, cv::Mat m);

void remove_haze(char* img_name, char* out_img_name);
cv::Mat remove_haze(cv::Mat img);

