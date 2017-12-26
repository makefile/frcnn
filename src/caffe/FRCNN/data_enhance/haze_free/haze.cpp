#define _CRT_SECURE_NO_WARNINGS
#include "haze.h"
#include "guidedfilter.h"
#include <cstdio>
//#pragma comment( lib, "opencv_world310d.lib" ) 
using namespace std;
using namespace cv;

//these var can be shared in thread
const int _PriorSize = 15;		//窗口大小
const double _topbright = 0.001;//亮度最高的像素比例
const double _w = 0.95;			//w
const float t0 = 0.1;			//T(x)的最小值   因为不能让tx小于0 等于0 效果不好
//int SizeH = 0;			//图片高度
//int SizeW = 0;			//图片宽度
//int SizeH_W = 0;			//图片中的像素总 数 H*W
//Vec3f a;//全球大气的光照值
//Mat trans_refine;
//Mat dark_out1;

//char img_name[100];//文件名

//读入图片
Mat ReadImage(char* img_name)
{

	Mat img = imread(img_name);

	//SizeH = img.rows;
	//SizeW = img.cols;
	//SizeH_W = img.rows*img.cols;

	Mat real_img(img.rows, img.cols, CV_32FC3);
	img.convertTo(real_img, CV_32FC3);

	real_img = real_img / 255;

	return real_img;

	//读入图片 并其转换为3通道的矩阵后 
	//除以255 将其RBG确定在0-1之间
}



//计算暗通道
//J^{dark}(x)=min( min( J^c(y) ) )
Mat DarkChannelPrior(Mat img, Mat &dark_out1)
{
	Mat dark = Mat::zeros(img.rows, img.cols, CV_32FC1);//新建一个所有元素为0的单通道的矩阵

	for (int i = 0; i<img.rows; i++)
	{
		for (int j = 0; j<img.cols; j++)
		{

			dark.at<float>(i, j) = min(
					min(img.at<Vec3f>(i, j)[0], img.at<Vec3f>(i, j)[1]),
					min(img.at<Vec3f>(i, j)[0], img.at<Vec3f>(i, j)[2])
					);//就是两个最小值的过程
		}
	}
	erode(dark, dark_out1, Mat::ones(_PriorSize, _PriorSize, CV_32FC1));//这个函数叫腐蚀 做的是窗口大小的模板运算 ,对应的是最小值滤波,即 黑色图像中的一块块的东西

	return dark_out1;//这里dark_out1用的是全局变量，因为在其它地方也要用到
}
Mat DarkChannelPrior_(Mat img, Vec3f a)//这个函数在计算tx用到，因为与计算暗通道一样都用到了求最小值的过程，变化不多，所以改了改就用这里了
{
	double A = (a[0] + a[1] + a[2]) / 3.0;//全球大气光照值 此处是3通道的平均值

	Mat dark = Mat::zeros(img.rows, img.cols, CV_32FC1);
	Mat dark_out = Mat::zeros(img.rows, img.cols, CV_32FC1);
	for (int i = 0; i<img.rows; i++)
	{
		for (int j = 0; j<img.cols; j++)
		{
			dark.at<float>(i, j) = min(
					min(img.at<Vec3f>(i, j)[0] / A, img.at<Vec3f>(i, j)[1] / A),
					min(img.at<Vec3f>(i, j)[0] / A, img.at<Vec3f>(i, j)[2] / A)
					);//同理


		}
	}
	erode(dark, dark_out, Mat::ones(_PriorSize, _PriorSize, CV_32FC1));//同上

	return dark_out;

}


//计算A的值
Vec3f Airlight(Mat img, Mat dark)//vec<float ,3>表示有3个大小的vector 类型为float
{
	int SizeH_W = img.rows * img.cols;
	int n_bright = _topbright*SizeH_W;

	Mat dark_1 = dark.reshape(1, SizeH_W);//这里dark_1是一个有图片像素那么多行的矩阵 方便下面循环计算

	vector<int> max_idx;

	float max_num = 0;

	Vec3f A(0, 0, 0);
	Mat RGBPixcels = Mat::ones(n_bright, 1, CV_32FC3);

	for (int i = 0; i<n_bright; i++)
	{
		max_num = 0;
		max_idx.push_back(max_num);
		for (float * p = (float *)dark_1.datastart; p != (float *)dark_1.dataend; p++)
		{
			if (*p>max_num)
			{
				max_num = *p;//记录光照的最大值

				max_idx[i] = (p - (float *)dark_1.datastart);//位置

				RGBPixcels.at<Vec3f>(i, 0) = ((Vec3f *)img.data)[max_idx[i]];//对应 的三个通道的值给RGBPixcels

			}
		}
		((float *)dark_1.data)[max_idx[i]] = 0;//访问过的标记为0，这样就不会重复访问
	}


	for (int j = 0; j<n_bright; j++)
	{

		A[0] += RGBPixcels.at<Vec3f>(j, 0)[0];
		A[1] += RGBPixcels.at<Vec3f>(j, 0)[1];
		A[2] += RGBPixcels.at<Vec3f>(j, 0)[2];

	}//将光照值累加

	A[0] /= n_bright;
	A[1] /= n_bright;
	A[2] /= n_bright;//除以总数   即取所有符合的点的平均值。

	return A;
}


//Calculate Transmission Matrix
Mat TransmissionMat(Mat dark, Mat &dark_out1, Vec3f a)
{
	double A = (a[0] + a[1] + a[2]) / 3.0;
	for (int i = 0; i < dark.rows; i++)
	{
		for (int j = 0; j < dark.cols; j++)
		{
			double temp = (dark_out1.at<float>(i, j));
			double B = fabs(A - temp);
			//	conut++;
			//cout << conut << endl;
			//if (B==)
			if (B - 0.3137254901960784 < 0.0000000000001)//K=80    80/255=0.31   这里浮点数要这样做减法才能正确的比较
			{
				dark.at<float>(i, j) = (1 - _w*dark.at<float>(i, j))*
					(0.3137254901960784 / (B));//此处为改过的式子部分
			}
			else
			{
				dark.at<float>(i, j) = 1 - _w*dark.at<float>(i, j);
			}
			if (dark.at<float>(i, j) <= 0.2)//保证Tx不失真，因为会以上除出的结果会有不对
			{
				dark.at<float>(i, j) = 0.5;
			}
			if (dark.at<float>(i, j) >= 1)//同上
			{
				dark.at<float>(i, j) = 1.0;
			}

		}
	}

	return dark;
}
Mat TransmissionMat1(Mat dark, Vec3f a)
{
	//double A = (a[0] + a[1] + a[2]) / 3.0;
	for (int i = 0; i < dark.rows; i++)
	{
		for (int j = 0; j < dark.cols; j++)
		{

			dark.at<float>(i, j) = (1 - _w*dark.at<float>(i, j));

		}
	}

	return dark;
}
//Calculate Haze Free Image
Mat hazefree(Mat img, Mat t, Vec3f a, float exposure = 0)//此处的exposure的值表示去雾后应该加亮的值。
{
	double AAA = a[0];
	if (a[1] > AAA)
		AAA = a[1];
	if (a[2] > AAA)
		AAA = a[2];
	//取a中的最大的值


	//新开一个矩阵
	Mat freeimg = Mat::zeros(img.rows, img.cols, CV_32FC3);
	img.copyTo(freeimg);

	//两个迭代器，这样的写法可以不用两层循环，比较快点
	Vec3f * p = (Vec3f *)freeimg.datastart;
	float * q = (float *)t.datastart;

	for (; p<(Vec3f *)freeimg.dataend && q<(float *)t.dataend; p++, q++)
	{
		(*p)[0] = ((*p)[0] - AAA) / std::max(*q, t0) + AAA + exposure;
		(*p)[1] = ((*p)[1] - AAA) / std::max(*q, t0) + AAA + exposure;
		(*p)[2] = ((*p)[2] - AAA) / std::max(*q, t0) + AAA + exposure;
	}

	return freeimg;
}


void printMatInfo(char * name, Mat m)
{
	cout << name << ":" << endl;
	cout << "\t" << "cols=" << m.cols << endl;
	cout << "\t" << "rows=" << m.rows << endl;
	cout << "\t" << "channels=" << m.channels() << endl;
}


//Main Function
//int main(int argc, char * argv[])
cv::Mat remove_haze(cv::Mat img) {
	//char filename[100];
	//sscanf(argv[1],"%s",img_name);
	//cin >> img_name;
	/*while (_access(img_name, 0) != 0)//检测图片是否存在
	  {
	  std::cout << "The image " << img_name << " don't exist." << endl << "Please enter another one:" << endl;
	//cin >> filename;
	}*/

	//clock_t start, finish;
	//double duration1, duration3, duration4, duration7;

	//读入图片
	//cout << "读入图片 ... " << img_name << endl;

	//start = clock();
	//Mat img = ReadImage(img_name);
	Mat src_img = img.clone();//I am not sure if img will change in later code,so I copy that.
	//imshow("原图", img);
	//printMatInfo("img", img);
	//finish = clock();
	//duration1 = (double)(finish - start) / CLOCKS_PER_SEC;
	//cout << "Time Cost: " << duration1 << "s" << endl;//输出这一步的时间
	//cout << endl;

	//计算暗通道
	//cout << "计算暗通道 ..." << endl;

	//start = clock();
	Mat dark_out1;
	Mat dark_channel = DarkChannelPrior(img, dark_out1);
	//imshow("Dark Channel Prior", dark_channel);
	//printMatInfo((char*)"dark_channel", dark_channel);
	//finish = clock();
	//duration3 = (double)(finish - start) / CLOCKS_PER_SEC;
	//cout << "Time Cost: " << duration3 << "s" << endl;
	//cout << endl;

	//计算全球光照值
	//cout << "计算A值 ..." << endl;
	//start = clock();
	Vec3f a = Airlight(img, dark_channel);
	//cout << "Airlight:\t" << " B:" << a[0] << " G:" << a[1] << " R:" << a[2] << endl;
	//finish = clock();
	//duration4 = (double)(finish - start) / CLOCKS_PER_SEC;
	//cout << "Time Cost: " << duration4 << "s" << endl;
	//cout << endl;

	//计算tx
	//cout << "Reading Refine Transmission..." << endl;
	Mat trans_refine = TransmissionMat(DarkChannelPrior_(src_img,a),dark_out1, a);
	//printMatInfo("trans_refine", trans_refine);
	//imshow("Refined Transmission Mat",trans_refine);
	//cout << endl;

	Mat tran = guidedFilter(img, trans_refine, 60, 0.0001);//导向滤波 得到精细的透射率图
	//imshow("fitler", tran);

	//去雾

	//cout << "Calculating Haze Free Image ..." << endl;
	//start = clock();
	Mat free_img = hazefree(img, tran, a, 0);//此处 如果用tran的话就是导向滤波部分
	//如果是trans_refine 就没有用导向滤波 效果不是那么						的好
	/*
	   上面第四个参数是用来增加亮度的，0.1比较好
	 */


	//finish = clock();
	//duration7 = (double)(finish - start) / CLOCKS_PER_SEC;
	//cout << "Time Cost: " << duration7 << "s" << endl;
	//cout << "Total Time Cost: " << duration1 + duration3 + duration4 + duration7 << "s" << endl;

//	imwrite(out_img_name, free_img * 255);
	//imwrite("output.jpg", free_img * 255);
	//imshow("去雾后", free_img);
	//waitKey();
	return free_img * 255;
}

void remove_haze(char* img_name, char* out_img_name) {
	Mat img = ReadImage(img_name);
	Mat result = remove_haze(img);
	imwrite(out_img_name, result);
}

