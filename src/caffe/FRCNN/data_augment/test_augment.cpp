
#include "data_utils.hpp"
using namespace cv;

void main_test_augment_and_rotate()
{
	//char *img_path = "E:/datasets/NWPU/ship/ship_022.jpg";
	string img_path = "test.jpg";
	float img_width = 256;
	float img_height = 256;
	int resize_h = 0;
	int resize_w = 0;
	int flip = 0;// rand() % 2;
	float jitter = 0.2;
	float hue = .1;
	float saturation = 1.5;// .75;
	float exposure = 1.5;//.75;
	set_rand_seed(-1);
	image orig = load_image_color(img_path.c_str(), 0, 0);
	
	std::vector<std::vector<float> > rois;
	std::vector<float> tmp{ 1, 80, 80, 120, 200 };//label x1 y1 x2 y2
	rois.push_back(tmp);
	int num_boxes = rois.size();
	box_label *boxes = (box_label*)calloc(num_boxes, sizeof(box_label));
	convert_box(rois, boxes, img_width, img_height);

	cv::Mat origmat = image2cvmat(orig);
	//	show_image(orig, "orig");
	cvDrawDottedRect(origmat, cv::Point(rois[0][1], rois[0][2]), cv::Point(rois[0][3], rois[0][4]), cv::Scalar(200, 0, 0), 6, 2);
	cv::imshow("origmat", origmat);

	// ratate, angle range(0,2*PI)
//	float angle = 3* M_PI / 4;//anti-clockwise direction
	float angle = M_PI_2;//anti-clockwise direction
	if (angle > 0)
	{
		box_label *boxes_new = (box_label*)calloc(num_boxes, sizeof(box_label));
		image rot = rotate_augment(angle, orig, boxes, boxes_new, num_boxes);
		//	show_image(rot, "rot");
		cv::Mat mat = image2cvmat(rot);
		std::vector<std::vector<float> > rois_new = convert_box(boxes_new, num_boxes, img_width, img_height);
		cvDrawDottedRect(mat, cv::Point(rois_new[0][1], rois_new[0][2]), cv::Point(rois_new[0][3], rois_new[0][4]), cv::Scalar(200, 0, 0), 6, 2);
//		cvDrawDottedRect(mat, cv::Point(rois[0][0], rois[0][1]), cv::Point(rois[0][2], rois[0][3]), cv::Scalar(200, 0, 0), 6, 2);
		cv::imshow("rot", mat);
	}

	//augment
	image result = data_augment(orig, boxes, num_boxes, resize_w, resize_h, flip, jitter, hue, saturation, exposure);
	rois = convert_box(boxes, num_boxes, img_width, img_height);
	free(boxes);
//	show_image(result, "result");
	cv::Mat mat = image2cvmat(result);
//	cvDrawDottedRect(mat, cv::Point(10, 10), cv::Point(90, 90), cv::Scalar(200, 0, 0), 6, 2);
//	std::cout << rois[0][0] << ' ' << rois[0][1] << ' ' << rois[0][2] << ' ' << rois[0][3] << std::endl;
	cvDrawDottedRect(mat, cv::Point(rois[0][1], rois[0][2]), cv::Point(rois[0][3], rois[0][4]), cv::Scalar(200, 0, 0), 6, 2);
//	cv::imshow ("dashed result", mat);
	cvWaitKey(0);
	free_image(orig);
}
