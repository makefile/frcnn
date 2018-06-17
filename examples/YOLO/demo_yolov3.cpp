#include <gflags/gflags.h>
#include <glog/logging.h>
#include "boost/algorithm/string.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/signal_handler.h"
//#include "caffe/FRCNN/util/frcnn_vis.hpp"
#include "api/api.hpp"
//for yolo v3
#include "caffe/YOLO/yolo_layer.h"
#include "caffe/YOLO/image.h"

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

  std::string image_dir = FLAGS_image_dir.c_str();
  std::string out_dir = FLAGS_out_dir.c_str();
  std::vector<std::string> images = caffe::Frcnn::get_file_list(image_dir, ".jpg");

//  API::Set_Config(default_config_file);
//  API::Detector detector(proto_file, model_file); 
  /* Load the network. */
  shared_ptr<Net<float> > net;
  net.reset(new Net<float>(proto_file, caffe::TEST));
  net->CopyTrainedLayersFrom(model_file);
  CHECK_EQ(net->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net->num_outputs(), 3) << "Network should have exactly three outputs.";  
  Blob<float> *input_data_blobs = net->input_blobs()[0];
    LOG(INFO) << "Input data layer channels is  " << input_data_blobs->channels();
    LOG(INFO) << "Input data layer width is  " << input_data_blobs->width();
    LOG(INFO) << "Input data layer height is  " << input_data_blobs->height();

  int input_size = input_data_blobs->channels()*input_data_blobs->width()*input_data_blobs->height();

  //std::vector<caffe::Frcnn::BBox<float> > results;
  caffe::Timer time_;
  DLOG(INFO) << "Test Image Dir : " << image_dir << "  , have " << images.size() << " pictures!";
  DLOG(INFO) << "Output Dir Is : " << out_dir;
  for (size_t index = 0; index < images.size(); ++index) {
    DLOG(INFO) << std::endl << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl
        << "Demo for " << images[index];
    cv::Mat img = cv::imread(image_dir+images[index]);
    //load image
    std::string im_path = image_dir + images[index];
    image im = load_image_color((char*)im_path.c_str(),0,0);
    image sized = letterbox_image(im,input_data_blobs->width(),input_data_blobs->height());
    //CHECK_EQ(input_size, sized.w * sized.h * sized.c);
    // reshape not always needed, but if you specify different size, you should do it
    input_data_blobs->Reshape(input_data_blobs->num(), input_data_blobs->channels(), input_data_blobs->width(), input_data_blobs->height());
    float *blob_data = input_data_blobs->mutable_cpu_data();
    std::memcpy(blob_data, sized.data, sizeof(float) * input_size);

    time_.Start();

    //detector.predict(image, results);
    // net->Forward(); //only pass once
    float loss;
    net->Forward(&loss); // thus can forward any times
    vector<Blob<float>*> blobs;
    Blob<float>* out_blob1 = net->output_blobs()[1];
    blobs.push_back(out_blob1);
    Blob<float>* out_blob2 = net->output_blobs()[2];
    blobs.push_back(out_blob2);
    Blob<float>* out_blob3 = net->output_blobs()[0];
    blobs.push_back(out_blob3);
    
    int classes = 80;
    float thresh = 0.5;
    float nms = 0.3;
    int nboxes = 0;
    detection *dets = get_detections(blobs,im.w,im.h,input_data_blobs->width(), input_data_blobs->height(), &nboxes, classes, thresh, nms);

    LOG(INFO) << "Predict " << images[index] << " cost " << time_.MilliSeconds() << " ms."; 
    //LOG(INFO) << "There are " <<  << " objects in picture.";
    int i,j;
    for(i=0;i< nboxes;++i){
        //char labelstr[4096] = {0};
        int cls = -1;
        for(j=0;j<80;++j){
            if(dets[i].prob[j] > 0.5){
                if(cls < 0){
                    cls = j;
                }
                printf("%d: %.0f%%\n",cls,dets[i].prob[j]*100);
            }
        }
        if(cls >= 0){
            box b = dets[i].bbox;
            printf("x = %f,y =  %f,w = %f,h =  %f\n",b.x,b.y,b.w,b.h);

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;
            cv::rectangle(img,cv::Point(left,top),cv::Point(right,bot),cv::Scalar(0,0,255),3,8,0);
            printf("left = %d,right =  %d,top = %d,bot =  %d\n",left,right,top,bot);
        }
    }
    
    std::string name = out_dir+images[index];
    cv::imwrite(name, img);
    //char xx[100];
    //sprintf(xx, "%s", name.c_str());
    //cv::imwrite(std::string(xx), img);

    // free memory
    free_detections(dets,nboxes);
    free_image(im);
    free_image(sized);
  }
  return 0;
}
