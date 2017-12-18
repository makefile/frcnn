#include <fstream>
#include <sstream>
#include "caffe/FRCNN/util/frcnn_vis.hpp"
using namespace cv;
namespace caffe {
    namespace Frcnn {
        /*!
         *  @brief 对浮点数四舍五入后指定位数输出
         *
         *  @param dbNum  [in]待处理的浮点数
         *  @param decplaces [in]小数点后的位数
         *  @return 输出字符串
         */
        std::string NumRounding(double dbNum,int decplaces)
        {
            // stringstream对象默认精度为,这里利用numeric_limits将精度设置为long double覆盖默认精度
            int prec=std::numeric_limits<long double>::digits10; 
            std::ostringstream oss;
            oss.precision(prec);
            oss<<dbNum;
            std::string strNum = oss.str();
            // 求取小数点位置
            size_t DecPos = strNum.find('.');
            // 若找不到小数点，就直接返回
            if(DecPos==std::string::npos)
                return strNum;
            //假如原有的小数点位数小于等于有效位数
            size_t len = strNum.size();
            if((len-DecPos-1)<=decplaces)
                return strNum;
            // 先进行四舍五入，比如输出四舍五入一位，就加.05
            int nTmp = decplaces+1;
            double exa = 1.0;
            while(nTmp--)
            {
                exa = exa/10.0;
            }
            double dbAppen = 5*exa;
            double tmp = dbNum + dbAppen;

            // 清空缓存，重新进行格式化
            oss.str(""); 
            oss<<tmp;
            std::string strResult = oss.str();
            // 截取字符串
            strResult = strResult.substr(0,strResult.find('.')+decplaces+1);
            return strResult;
        }
        /* this is for reference
         * http://blog.csdn.net/a553654745/article/details/45743063
         * https://docs.opencv.org/3.1.0/db/dd6/classcv_1_1RotatedRect.html
         void RotatedRect::points(Point2f pt[]) const  
         {  
         double _angle = angle*CV_PI/180.;  
         float b = (float)cos(_angle)*0.5f;  
         float a = (float)sin(_angle)*0.5f;  

         pt[0].x = center.x - a*size.height - b*size.width;  
         pt[0].y = center.y + b*size.height - a*size.width;  
         pt[1].x = center.x + a*size.height - b*size.width;  
         pt[1].y = center.y - b*size.height - a*size.width;  
         pt[2].x = 2*center.x - pt[0].x;  
         pt[2].y = 2*center.y - pt[0].y;  
         pt[3].x = 2*center.x - pt[1].x;  
         pt[3].y = 2*center.y - pt[1].y;  
         } 
         */
        template <typename Dtype>
            void draw_poly(Mat& img, const RBBox<Dtype>& b, std::string text) {
                RotatedRect rRect = RotatedRect(Point2f(b[0],b[1]), Size2f(b[2],b[3]), - b[4] * 180.0/M_PI);// the angle is clockwise direction
                Point2f vertices[4];
                rRect.points(vertices);
                Point2f pc[2];
                for (int i = 0; i < 4; i++) {
                    line(img, vertices[i], vertices[(i+1)%4], Scalar(0,255,0));
                    if (i  == 0) {
                        float xc1 = (vertices[i].x + vertices[(i+1)%4].x) / 2;
                        float yc1 = (vertices[i].y + vertices[(i+1)%4].y) / 2;
                        pc[0] = Point2f(xc1,yc1);
                        cv::circle(img,pc[0],6,cv::Scalar(0,255,255));
                    }
                    else if (i  == 2) {
                        float xc2 = (vertices[i].x + vertices[(i+1)%4].x) / 2;
                        float yc2 = (vertices[i].y + vertices[(i+1)%4].y) / 2;
                        pc[1] = Point2f(xc2,yc2);
                        cv::circle(img,pc[1],6,cv::Scalar(0,255,255));
                    }
                }
                Rect brect = rRect.boundingRect();
                rectangle(img, brect, Scalar(0,0,255));
                //rectangle(img, pc[0],pc[1], Scalar(0,0,255));
                //cv::putText(img, text, cv::Point(pc[0].x,pc[0].y - 8), cv::FONT_HERSHEY_COMMPLEX, 0.6, cv::Scalar(0,255,0));
                cv::putText(img, text, cv::Point(brect.x, brect.y - 8), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0,255,0));
                /*
                   Dtype w1 = b[3] * fabs(sin(b[4]));
                   Dtype w2 = b[2] * fabs(cos(b[4]));
                   Dtype h1 = b[3] * fabs(cos(b[4]));
                   Dtype h2 = b[2] * fabs(sin(b[4]));
                   Dtype x1 = b[0] - (w1 + w2 )/2;
                   Dtype y1 = b[1] + (h1 - h2)/2;
                   Dtype x2 = b[0] + (w2 - w1)/2;
                   Dtype y2 = b[1] + (h1 + h2)/2;
                   Dtype x3 = b[0] + (w1 + w2)/2;
                   Dtype y3 = y2 - h1;
                   Dtype x4 = x1 + w1;
                   Dtype y4 = b[1] - (h1 + h2)/2;
                   Point points[1][4];
                   points[0][0] = Point( x1, y1 );
                   points[0][1] = Point( x2, y2 );
                   points[0][2] = Point( x3, y3 );
                   points[0][3] = Point( x4, y4 );
                   const Point* pt[1] = { points[0] }; 
                   int npt[1] = {4};
                   polylines( img, pt, npt, 1, 1, Scalar(0,250,0), 3);//thickness=3 
                //https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#polylines
                */
            }
        template void draw_poly(Mat& img, const RBBox<float>& b,  string text);
        template void draw_poly(Mat& img, const RBBox<double>& b, string text);

        template <typename Dtype>
            void vis_detections(cv::Mat & frame, const std::vector<RBBox<Dtype> >& ans, const std::map<int,std::string> CLASS) { 
                for(size_t i = 0 ; i < ans.size() ; i++) {
                    //cv::rectangle(frame, cv::Point(ans[i][0],ans[i][1]) , cv::Point(ans[i][2],ans[i][3]) , cv::Scalar(255,255,255) );
                    std::ostringstream text;
                    text << GetClassName(CLASS, ans[i].id) << ":" << NumRounding(ans[i].confidence,2);
                    //cv::putText(frame, text.str() , cv::Point(ans[i][0]-12,ans[i][1]) , 0 , 0.6 , cv::Scalar(0,255,0) );
                    draw_poly(frame, (RBBox<Dtype>)ans[i], text.str());
                }
            }

        template void vis_detections(cv::Mat & frame, const std::vector<RBBox<float> >& ans, const std::map<int,std::string> CLASS);
        template void vis_detections(cv::Mat & frame, const std::vector<RBBox<double> >& ans, const std::map<int,std::string> CLASS);

        template <typename Dtype>
            void vis_detections(cv::Mat & frame, const RBBox<Dtype> ans, const std::map<int,std::string> CLASS) { 
                std::vector<RBBox<Dtype> > vec_ans;
                vec_ans.push_back( ans );
                vis_detections(frame, vec_ans, CLASS);
            }

        template void vis_detections(cv::Mat & frame, const RBBox<float> ans, const std::map<int,std::string> CLASS);
        template void vis_detections(cv::Mat & frame, const RBBox<double> ans, const std::map<int,std::string> CLASS);

    } // Frcnn

} // caffe 
