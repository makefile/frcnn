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
DEFINE_string(image_dir, "",
        "Optional;Test images Dir."); 
DEFINE_string(out_dir, "",
        "Optional;Output images Dir."); 
DEFINE_double(thresh, 0.0, "Optional;confidence thresh.");

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
            "  --image_dir    file    input image dir \n"
            "  --thresh       0.0     confidence thresh for result \n"  
            "  --out_dir      file    output image dir ");
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
    //fyk
    std::vector<std::string> cls {"1", "carrier", "war", "mer"};
    std::map<int,std::string> cls_map = caffe::Frcnn::load_class_map(cls);
    std::string image_dir = FLAGS_image_dir.c_str();
    std::string out_dir = FLAGS_out_dir.c_str();
    double thresh = FLAGS_thresh;
    //std::vector<std::string> images = caffe::Frcnn::get_file_list(image_dir, ".jpg");
    std::vector<std::string> images = caffe::Frcnn::get_file_list(image_dir, "");

    API::Set_Config(default_config_file);
    API::Detector detector(proto_file, model_file); 

    std::vector<caffe::Frcnn::RBBox<float> > results;
    caffe::Timer time_;
    DLOG(INFO) << "Test Image Dir : " << image_dir << "  , have " << images.size() << " pictures!";
    DLOG(INFO) << "Output Dir Is : " << out_dir;
    for (size_t index = 0; index < images.size(); ++index) {
        DLOG(INFO) << std::endl << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl
            << "Demo for " << images[index];
        cv::Mat image = cv::imread(image_dir+images[index]);
        time_.Start();
        detector.predict(image, results);
        LOG(INFO) << "Predict " << images[index] << " cost " << time_.MilliSeconds() << " ms."; 
        LOG(INFO) << "There are " << results.size() << " objects in picture.";
        /*for (size_t obj = 0; obj < results.size(); obj++) {
            LOG(INFO) << results[obj].to_string();
        }*/
        cv::Mat ori ; 
        image.convertTo(ori, CV_32FC3);
        for (int label = 0; label < caffe::Frcnn::FrcnnParam::n_classes; label++) {
            std::vector<caffe::Frcnn::RBBox<float> > cur_res;
            for (size_t idx = 0; idx < results.size(); idx++) {
                if (results[idx].id == label) {
                    if (results[idx].confidence > thresh) cur_res.push_back( results[idx] );
                }
            }
            if (cur_res.size() == 0) continue;
            caffe::Frcnn::vis_detections(ori, cur_res, cls_map);
        }
        std::string name = out_dir + "/" + images[index];
        //char out_name[100];
        //sprintf(out_name, "%s_%s.jpg", name.c_str(), caffe::Frcnn::GetClassName(caffe::Frcnn::LoadClass(cls),label).c_str());
        //sprintf(out_name, "%s_infer.jpg", name.c_str());
        ori.convertTo(ori, CV_8UC3);
        cv::imwrite(name.replace(name.rfind("."),10,"_infer.jpg"), ori);//rfind is reverse find
        //cv::imshow( "detection", ori );
        //cv::waitKey(0);
    }
    return 0;
}
