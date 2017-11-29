#ifndef __EQ_HIST__
#define __EQ_HIST__

#include <opencv2/opencv.hpp>
using namespace cv;
Mat equalizeChannelHist(const Mat & inputImage);
Mat equalizeIntensityHist(const Mat & inputImage);

#endif
