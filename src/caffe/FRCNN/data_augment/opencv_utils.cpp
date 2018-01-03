#include "data_utils.hpp"
using namespace cv;

/*
// drawDottedRect for drawing dotted lines and rectangle.
// Maxime Tremblay, 2010, Universit¨¦ Laval, Qu¨¦bec city, QB, Canada 
// http://opencv-users.1802565.n2.nabble.com/drawDottedRect-td5661156.html
*/

void cvDrawDottedLine(IplImage* img, CvPoint pt1, CvPoint pt2, CvScalar color, int thickness, int lenghOfDots, int lineType, int leftToRight)
{
	CvLineIterator iterator;
	int count = cvInitLineIterator(img, pt1, pt2, &iterator, lineType, leftToRight);
	int offset, x, y;


	for (int i = 0; i < count; i = i + (lenghOfDots * 2 - 1))
	{
		if (i + lenghOfDots > count)
			break;

		offset = iterator.ptr - (uchar*)(img->imageData);
		y = offset / img->widthStep;
		x = (offset - y*img->widthStep) / (3 * sizeof(uchar) /* size of pixel */);

		CvPoint lTemp1 = cvPoint(x, y);
		for (int j = 0; j<lenghOfDots - 1; j++)	//I want to know have the last of these in the iterator 
			CV_NEXT_LINE_POINT(iterator);

		offset = iterator.ptr - (uchar*)(img->imageData);
		y = offset / img->widthStep;
		x = (offset - y*img->widthStep) / (3 * sizeof(uchar) /* size of pixel */);

		CvPoint lTemp2 = cvPoint(x, y);
		cvDrawLine(img, lTemp1, lTemp2, color, thickness, lineType);
		for (int j = 0; j<lenghOfDots; j++)
			CV_NEXT_LINE_POINT(iterator);
	}
}

void cvDrawDottedRect(IplImage* img, CvPoint pt1, CvPoint pt2, CvScalar color, int thickness, int lenghOfDots, int lineType)
{	//1---2 
	//|	  | 
	//4---3 
	//	1 --> pt1, 2 --> tempPt1, 3 --> pt2, 4 --> tempPt2 

	CvPoint tempPt1 = cvPoint(pt2.x, pt1.y);
	CvPoint tempPt2 = cvPoint(pt1.x, pt2.y);
	cvDrawDottedLine(img, pt1, tempPt1, color, thickness, lenghOfDots, lineType, 0);
	cvDrawDottedLine(img, tempPt1, pt2, color, thickness, lenghOfDots, lineType, 0);
	cvDrawDottedLine(img, pt2, tempPt2, color, thickness, lenghOfDots, lineType, 1);
	cvDrawDottedLine(img, tempPt2, pt1, color, thickness, lenghOfDots, lineType, 1);
}
void cvDrawDottedRect(Mat& mat_img, Point pt1, Point pt2, Scalar color, int lenghOfDots, int thickness, int lineType)
{	//1---2 
	//|	  | 
	//4---3 
	//	1 --> pt1, 2 --> tempPt1, 3 --> pt2, 4 --> tempPt2 
	Mat mat = mat_img;
	if (mat_img.type() != CV_8UC3)
	{
		mat_img.convertTo(mat, CV_8UC3);
	}
	IplImage ix = mat;
	IplImage* img = &ix; //share data,only copy header
	CvPoint tempPt1 = cvPoint(pt2.x, pt1.y);
	CvPoint tempPt2 = cvPoint(pt1.x, pt2.y);
	cvDrawDottedLine(img, pt1, tempPt1, color, thickness, lenghOfDots, lineType, 0);
	cvDrawDottedLine(img, tempPt1, pt2, color, thickness, lenghOfDots, lineType, 0);
	cvDrawDottedLine(img, pt2, tempPt2, color, thickness, lenghOfDots, lineType, 1);
	cvDrawDottedLine(img, tempPt2, pt1, color, thickness, lenghOfDots, lineType, 1);
	mat_img = mat;
}
void cvDrawDottedRect(IplImage* img, CvRect rect, CvScalar color, int lenghOfDots, int thickness, int lineType)
{	//1---2 
	//|	  | 
	//4---3 
	CvPoint pt1 = cvPoint(rect.x, rect.y);
	CvPoint pt2 = cvPoint(rect.x + rect.width, rect.y);
	CvPoint pt3 = cvPoint(rect.x + rect.width, rect.y + rect.height);
	CvPoint pt4 = cvPoint(rect.x, rect.y + rect.height);

	cvDrawDottedLine(img, pt1, pt2, color, thickness, lenghOfDots, lineType, 0);
	cvDrawDottedLine(img, pt2, pt3, color, thickness, lenghOfDots, lineType, 0);
	cvDrawDottedLine(img, pt3, pt4, color, thickness, lenghOfDots, lineType, 1);
	cvDrawDottedLine(img, pt4, pt1, color, thickness, lenghOfDots, lineType, 1);
}
