#include <gflags/gflags.h>
#include <glog/logging.h>
#include "boost/algorithm/string.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/FRCNN/util/frcnn_vis.hpp"
#include "api/api.hpp"

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

inline int INT(float x) { return int(x); };
inline std::string FloatToString(float x) { char A[100]; sprintf(A,"%.8f",x); return std::string(A);};

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

  std::string image_list = FLAGS_image_list.c_str();
  std::string image_root = FLAGS_image_root.c_str();
  std::string out_file = FLAGS_out_file.c_str();

  API::Set_Config( default_config_file );
  API::Rpn_Det detector(proto_file, model_file);
  std::vector<caffe::Frcnn::BBox<float> > results;

  LOG(INFO) << "image list is  : " << image_list;
  LOG(INFO) << "output file is : " << out_file;
  std::ifstream infile(image_list.c_str());
  std::ofstream otfile(out_file.c_str());
  API::DataPrepare data_load;
  int count = 0;
  while ( data_load.load_WithDiff(infile) ) {
    std::string image = data_load.GetImagePath("");
    cv::Mat cv_image = cv::imread(image_root+image);
    detector.predict(cv_image, results);
    otfile << "# " << data_load.GetImageIndex() << std::endl;
    otfile << image << std::endl;
    otfile << results.size() << std::endl;
    for (size_t obj = 0; obj < results.size(); obj++) {
      otfile << results[obj].id << "  " << INT(results[obj][0]) << " " << INT(results[obj][1]) << " " << INT(results[obj][2]) << " " << INT(results[obj][3]) << "     " << FloatToString(results[obj].confidence) << std::endl;
    }
    LOG(INFO) << "Handle " << ++count << " th image : " << image << " , Left " << results.size() << " proposals";
  }
  infile.close();
  otfile.close();
  return 0;
}
