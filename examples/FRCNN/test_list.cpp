/**
	Get inference result of image list.
	Input  format: list of image names
	Output format: image name,class_name,x1 y1 x2 y2
*/
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "boost/algorithm/string.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/FRCNN/util/frcnn_vis.hpp"
#include "caffe/FRCNN/util/equalize_hist.hpp"
#include "api/api.hpp"
#include <chrono>
using caffe::Frcnn::FrcnnParam;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on the given device ID, Empty is CPU");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "",
    "Trained Model By Faster RCNN End-to-End Pipeline.");
DEFINE_string(default_c, "",
    "Default config file path.");
DEFINE_string(image_list, "",
    "Optional;Test images list.");
DEFINE_string(image_root, "",
    "Optional;Test images root directory.");
DEFINE_string(out_file, "",
    "Optional;Output images file.");
DEFINE_string(submit_file, "",
    "Optional;submit file.");
DEFINE_int32(max_per_image, 100,
    "Limit to max_per_image detections *over all classes*");

//inline std::string INT(float x) { char A[100]; sprintf(A,"%.1f",x); return std::string(A);};
inline int INT(float x) { return (int)x;};//if x>=0,same as floor
inline std::string FloatToString(float x) { char A[100]; sprintf(A,"%.4f",x); return std::string(A);};

int main(int argc, char** argv){
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: demo_frcnn_api <args>\n\n"
      "args:\n"
      "  --gpu          7       use 7-th gpu device, default is cpu model\n"
      "  --model        file    protocol buffer text file\n"
      "  --weights      file    Trained Model\n"
      "  --default_c    file    Default Config File\n"
      "  --image_list   file    input image list\n"
      "  --image_root   file    input image dir\n"
      "  --max_per_image   file limit to max_per_image detections\n"
      "  --out_file     file    output amswer file");

  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  CHECK( FLAGS_gpu.size() == 0 || FLAGS_gpu.size() == 1 || (FLAGS_gpu.size()==2&&FLAGS_gpu=="-1")) << "Can only support one gpu or none or -1(for cpu)";
  int gpu_id = -1;
  if( FLAGS_gpu.size() > 0 )
    gpu_id = boost::lexical_cast<int>(FLAGS_gpu);

  if (gpu_id >= 0) {
#ifndef CPU_ONLY
    caffe::Caffe::SetDevice(gpu_id);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
#else
    LOG(FATAL) << "CPU ONLY MODEL, BUT PROVIDE GPU ID";
#endif
  } else {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }

  std::string proto_file             = FLAGS_model.c_str();
  std::string model_file             = FLAGS_weights.c_str();
  std::string default_config_file    = FLAGS_default_c.c_str();

  const std::string image_list = FLAGS_image_list.c_str();
  const std::string image_root = FLAGS_image_root.c_str();
  const std::string out_file = FLAGS_out_file.c_str();
  const std::string submit_file = FLAGS_submit_file.c_str();

  const int max_per_image = FLAGS_max_per_image;

  API::Set_Config(default_config_file);
  API::Detector detector(proto_file, model_file);

  LOG(INFO) << "image list     : " << image_list;
  LOG(INFO) << "output file    : " << out_file;
  LOG(INFO) << "max_per_image  : " << max_per_image;
  std::ifstream infile(image_list.c_str());
  std::ofstream otfile(out_file.c_str());
  std::ofstream submitfile(submit_file.c_str());
  API::DataPrepare data_load;
  int count = 0;
  std::string shot_dir, fname;
  std::string HASH;
  //int ids_;
  //float minx, miny, w,h;
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::string class_name[] = { "__background__", "huochuan", "youlun", "youting" };
  while ( infile >> fname ) {
    otfile << "# " << count << std::endl;
    for (int ii = 0; ii < 1; ii++) {
      //CHECK(fname.find(".jpg") != std::string::npos);
      //caffe::Frcnn::BBox<float> truth(minx, miny, minx + w, miny+h);
      //      cv::Mat cv_image = cv::imread(image_root + "/" + shot_dir + "/" + image);
      cv::Mat cv_image = cv::imread(image_root+fname);
      //fyk : do equlize_hist,only for 3-channel
      int he_case = FrcnnParam::use_hist_equalize; 
      switch(he_case){
      case 1:
      	cv_image = equalizeIntensityHist(cv_image); 
      	break;
      case 2:
      	cv_image = equalizeChannelHist(cv_image);
      	break;
      }
      //fyk end
      std::vector<caffe::Frcnn::BBox<float> > results;
      start = std::chrono::steady_clock::now();
      detector.predict(cv_image, results);
      end = std::chrono::steady_clock::now();
      float image_thresh = 0;
      if ( max_per_image > 0 ) {
        std::vector<float> image_score ;
        for (size_t obj = 0; obj < results.size(); obj++) {
          image_score.push_back(results[obj].confidence) ;
        }
        std::sort(image_score.begin(), image_score.end(), std::greater<float>());
        if ( max_per_image > image_score.size() ) {
          if ( image_score.size() > 0 )
            image_thresh = image_score.back();
        } else {
          image_thresh = image_score[max_per_image-1];
        }
      }
      std::vector<caffe::Frcnn::BBox<float> > filtered_res;

      for (size_t obj = 0; obj < results.size(); obj++) {
        if ( results[obj].confidence >= image_thresh ) {
          filtered_res.push_back( results[obj] );
        }
      }
//      caffe::Frcnn::BBox<float> best_bb = filtered_res[0];
//      float iou = caffe::Frcnn::get_iou(best_bb, truth);

//      std::cout<<"Finished image "<<fname<<" with intersection "<<iou<<" in "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<<" ms"<<std::endl;
//      const int ori_res_size = results.size();
      results = filtered_res;
      otfile << results.size() << std::endl;
	std::cout << count << "\tobjs:\t" << results.size() << std::endl;
      for (size_t obj = 0; obj < results.size(); obj++) {
        int x1 = INT(results[obj][0]);int y1 = INT(results[obj][1]);
        int x2 = INT(results[obj][2]);int y2 = INT(results[obj][3]);
	otfile << results[obj].id << "  " << x1 << " " << y1 << " " << x2 << " " << y2 << "     " << FloatToString(results[obj].confidence) << std::endl;
        //check if near the edge
	int gap = 3;
	if ( x1 < gap || y1 < gap || x2 > cv_image.cols - gap || y2 > cv_image.rows - gap ) {
		std::cout << "WARNING: near edge " << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;
		//continue;
	}
        submitfile << fname << "," << class_name[results[obj].id] << "," << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;
      }
    }
    count ++;
  }
  infile.close();
  otfile.close();
  return 0;
}
