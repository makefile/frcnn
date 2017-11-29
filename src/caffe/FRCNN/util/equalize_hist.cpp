#include "equalize_hist.hpp"
#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

Mat getGrayEqualHist(const Mat & src)
{
	Mat out;
	src.convertTo(out, CV_8UC1);
	return out;
}
Mat equalizeChannelHist(const Mat & inputImage)  
{
	int chs = inputImage.channels();
    if( chs >= 3 )  
    {  
        vector<Mat> channels;  
        split(inputImage, channels);  

        Mat B,G,R;  

        equalizeHist( channels[0], B );  
        equalizeHist( channels[1], G );  
        equalizeHist( channels[2], R );  

        vector<Mat> combined;  
        combined.push_back(B);  
        combined.push_back(G);  
        combined.push_back(R);  

        Mat result;  
        merge(combined, result);  

        return result;  
    }else if(chs==1) return getGrayEqualHist(inputImage);

    return inputImage;
    //return Mat();  
}
Mat equalizeIntensityHist(const Mat & inputImage)  
{  
	int chs = inputImage.channels();
    if( chs >= 3)  
    {  
        Mat ycrcb;  

        cvtColor(inputImage, ycrcb, COLOR_BGR2YCrCb);  

        vector<Mat> channels;  
        split(ycrcb, channels);  

        equalizeHist(channels[0], channels[0]);  

        Mat result;  
        merge(channels,ycrcb);  

        cvtColor(ycrcb, result, COLOR_YCrCb2BGR);  

        return result;  
    }else if(chs==1) return getGrayEqualHist(inputImage);

    return inputImage;
    //return Mat();  
}  
/*
int test_main( int argc, char** argv )
{
    Mat src, dst;

    string source_window = "Source image";
    string equalized_window = "Equalized Image";
    //src = imread( argv[1], 1 );
    src = imread( argv[1] );
    namedWindow( source_window, CV_WINDOW_AUTOSIZE );
    imshow( source_window, src );

    if( !src.data )
    { cout<<"Usage: ./Histogram_Demo <path_to_image>"<<endl;
        return -1;}

    Mat channel_color_dst = equalizeChannelHist(src);
    namedWindow( "channel", CV_WINDOW_AUTOSIZE );
    imshow( "channel", channel_color_dst );

    //intensity
    Mat intensity_color_dst = equalizeIntensityHist(src);
    namedWindow( "intensity", CV_WINDOW_AUTOSIZE );
    imshow( "intensity", intensity_color_dst );

    //gray
    cvtColor( src, src, CV_BGR2GRAY );
    equalizeHist( src, dst );

    namedWindow( equalized_window, CV_WINDOW_AUTOSIZE );
    imshow( equalized_window, dst );

    waitKey(0);

    return 0;
}
*/
