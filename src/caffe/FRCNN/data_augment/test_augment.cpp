
#include "data_utils.hpp"
using namespace cv;
std::vector<std::vector<float> > get_rois()
{
	std::vector<float> tmp{ 
		4, 202, 139, 254, 246,
		4, 268, 146, 316, 303,
		4, 329, 151, 366, 295,
		4, 381, 148, 411, 291,
		4, 422, 150, 454, 282,
		4, 466, 155, 491, 274 };//label x1 y1 x2 y2
	std::vector<std::vector<float> > rois; 
	for (int i = 0; i < tmp.size(); i += 5)
	{
		std::vector<float> tmp1 = { tmp[i], tmp[i + 1], tmp[i + 2], tmp[i + 3], tmp[i + 4] };
		rois.push_back(tmp1);
	}
	return rois;
}
void vis_32f_mat(std::string winname, cv::Mat mat)
{
	cv::Mat tmp_mat;
	mat.convertTo(tmp_mat, CV_8UC3);
	cv::imshow(winname, tmp_mat);
}
void print_rois(std::vector<std::vector<float> > rois)
{
	for (int i = 0; i < rois.size(); i++)
	{
		std::cout << rois[i][0] << ' ' << rois[i][1] << ' ' << rois[i][2] << ' ' << rois[i][3] << ' ' << rois[i][4] << std::endl;
	}
	std::cout << std::endl;
}
void test_main()
{
	char *img_path = "test.jpg";
	
	int flip = 1;// rand() % 2;
	float jitter = 0.2;
	float hue = .1;
	float saturation = 1.5;// .75;
	float exposure = 1.5;//.75;
	set_rand_seed(-1);
	image orig = load_image_color(img_path, 0, 0);
	float img_width = orig.w;
	float img_height = orig.h;

	std::vector<std::vector<float> > rois = get_rois();
	print_rois(rois);
	int num_boxes = rois.size();
	box_label *boxes = (box_label*)calloc(num_boxes, sizeof(box_label));
	convert_box(rois, boxes, img_width, img_height);

	cv::Mat origmat = image2cvmat(orig);
	//	show_image(orig, "orig");
	for (int i = 0; i < rois.size(); i++)
	{
		cvDrawDottedRect(origmat, cv::Point(rois[i][1], rois[i][2]), cv::Point(rois[i][3], rois[i][4]), cv::Scalar(200, 0, 0), 6, 1);
	}
	vis_32f_mat("orig", origmat);

	// ratate, angle range(0,2*PI)
//	float angle = 3* M_PI / 4;//anti-clockwise direction
	float angle = - M_PI_2;
//	float angle = M_PI;// M_PI_2;//anti-clockwise direction
	if (angle != 0)
	{
		box_label *boxes_new = (box_label*)calloc(num_boxes, sizeof(box_label));
		image rot = rotate_augment(angle, orig, boxes, boxes_new, num_boxes);
		//	show_image(rot, "rot");
		cv::Mat mat = image2cvmat(rot);
		std::vector<std::vector<float> > rois_new = convert_box(boxes_new, num_boxes, rot.w, rot.h);
		print_rois(rois_new);
		for (int i = 0; i < rois_new.size(); i++)
		{
			cvDrawDottedRect(mat, cv::Point(rois_new[i][1], rois_new[i][2]), cv::Point(rois_new[i][3], rois_new[i][4]), cv::Scalar(200, 0, 0), 6, 1);
		}
		vis_32f_mat("rot", mat);
	}

	//augment
	Mat result = data_augment(origmat, rois, flip, jitter, hue, saturation, exposure);
	print_rois(rois);
	free(boxes);
	for (int i = 0; i < rois.size(); i++)
	{
		cvDrawDottedRect(result, cv::Point(rois[i][1], rois[i][2]), cv::Point(rois[i][3], rois[i][4]), cv::Scalar(0, 0, 200), 6, 1);
	}
	vis_32f_mat("aug", result);
	cvWaitKey(0);
	free_image(orig);
}
