#include "api/FRCNN/frcnn_api.hpp"
#include "caffe/FRCNN/util/frcnn_gpu_nms.hpp"

namespace FRCNN_API{

using namespace caffe::Frcnn;

void Detector::predict_cascade(const cv::Mat &img_in, std::vector<std::vector<caffe::Frcnn::BBox<float> > > &results) {

  CHECK(FrcnnParam::test_scales.size() == 1) << "Only single-image batch implemented";

  float scale_factor = caffe::Frcnn::get_scale_factor(img_in.cols, img_in.rows, FrcnnParam::test_scales[0], FrcnnParam::test_max_size);

  cv::Mat img;
  const int height = img_in.rows;
  const int width = img_in.cols;
  DLOG(INFO) << "height: " << height << " width: " << width;
  img_in.convertTo(img, CV_32FC3);
  for (int r = 0; r < img.rows; r++) {
    for (int c = 0; c < img.cols; c++) {
      int offset = (r * img.cols + c) * 3;
      reinterpret_cast<float *>(img.data)[offset + 0] -= this->mean_[0]; // B
      reinterpret_cast<float *>(img.data)[offset + 1] -= this->mean_[1]; // G
      reinterpret_cast<float *>(img.data)[offset + 2] -= this->mean_[2]; // R
    }
  }
  //cv::resize(img, img, cv::Size(), scale_factor, scale_factor);
  //fyk: check decimation or zoom,use different method
  if( scale_factor < 1 )
    cv::resize(img, img, cv::Size(), scale_factor, scale_factor, cv::INTER_AREA);
  else
    cv::resize(img, img, cv::Size(), scale_factor, scale_factor);
  if (FrcnnParam::im_size_align > 0) {
    // pad to align im_size_align
    int new_im_height = int(std::ceil(img.rows / float(FrcnnParam::im_size_align)) * FrcnnParam::im_size_align);
    int new_im_width = int(std::ceil(img.cols / float(FrcnnParam::im_size_align)) * FrcnnParam::im_size_align);
    cv::Mat padded_im = cv::Mat::zeros(cv::Size(new_im_width, new_im_height), CV_32FC3);
    float *res_mat_data = (float *)img.data;
    float *new_mat_data = (float *)padded_im.data;
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            for (int k = 0; k < 3; ++k)
                new_mat_data[(y * new_im_width + x) * 3 + k] = res_mat_data[(y * img.cols + x) * 3 + k];
    img = padded_im;
  }

  std::vector<float> im_info(3);
  im_info[0] = img.rows;
  im_info[1] = img.cols;
  im_info[2] = scale_factor;

  DLOG(ERROR) << "im_info : " << im_info[0] << ", " << im_info[1] << ", " << im_info[2];
  this->preprocess(img, 0);
  this->preprocess(im_info, 1);

  std::string _blob_names[] = {"rois", "rois_2nd", "rois_3rd", "rois_2nd", "rois_3rd",
    "cls_prob", "cls_prob_2nd", "cls_prob_3rd", "cls_prob_2nd_avg", "cls_prob_3rd_avg",
    "bbox_pred", "bbox_pred_2nd", "bbox_pred_3rd", "bbox_pred_2nd", "bbox_pred_3rd"};
  int num_outputs = 5;
  vector<std::string> blob_names(_blob_names, _blob_names + 3 * num_outputs);
  static float zero_means[] = {0.0, 0.0, 0.0, 0.0};
  static float one_stds[] = {1.0, 1.0, 1.0, 1.0};
  static float cascade_stds[][4] = {0.1, 0.1, 0.2, 0.2,
                            0.05, 0.05, 0.1, 0.1,
                            0.033, 0.033, 0.067, 0.067};

  vector<boost::shared_ptr<Blob<float> > > output = this->predict(blob_names);
  for (int out_idx = 0; out_idx < num_outputs; out_idx++) {
    int stage = out_idx < 3 ? out_idx : out_idx-2;
    boost::shared_ptr<Blob<float> > rois(output[out_idx]);
    boost::shared_ptr<Blob<float> > cls_prob(output[num_outputs + out_idx]);
    boost::shared_ptr<Blob<float> > bbox_pred(output[2 * num_outputs + out_idx]);

    const int box_num = bbox_pred->num();
    const int cls_num = cls_prob->channels();
    CHECK_EQ(cls_num , caffe::Frcnn::FrcnnParam::n_classes);
    results[out_idx].clear();

    float* means = FrcnnParam::bbox_normalize_targets ? FrcnnParam::bbox_normalize_means : zero_means;
    //float* stds  = FrcnnParam::bbox_normalize_targets ? FrcnnParam::bbox_normalize_stds : one_stds;
    float* stds  = FrcnnParam::bbox_normalize_targets ? (float*)&cascade_stds[stage] : one_stds;
    for (int cls = 1; cls < cls_num; cls++) { 
      vector<BBox<float> > bbox;
      for (int i = 0; i < box_num; i++) { 
        float score = cls_prob->cpu_data()[i * cls_num + cls];
        // fyk: speed up
        if (score < FrcnnParam::test_score_thresh) continue;

        Point4f<float> roi(rois->cpu_data()[(i * 5) + 1]/scale_factor,
                       rois->cpu_data()[(i * 5) + 2]/scale_factor,
                       rois->cpu_data()[(i * 5) + 3]/scale_factor,
                       rois->cpu_data()[(i * 5) + 4]/scale_factor);

        Point4f<float> delta(bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 0] * stds[0] + means[0],
                       bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 1] * stds[1] + means[1],
                       bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 2] * stds[2] + means[2],
                       bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 3] * stds[3] + means[3]);

        Point4f<float> box = caffe::Frcnn::bbox_transform_inv(roi, delta);
        //fyk clip predicted boxes to image
        box[0] = std::max(0.0f, std::min(box[0], width - 1.f));
        box[1] = std::max(0.0f, std::min(box[1], height - 1.f));
        box[2] = std::max(0.0f, std::min(box[2], width - 1.f));
        box[3] = std::max(0.0f, std::min(box[3], height - 1.f));

        // BBox tmp(box, score, cls);
        // LOG(ERROR) << "cls: " << tmp.id << " score: " << tmp.confidence;
        // LOG(ERROR) << "roi: " << roi.to_string();
        bbox.push_back(BBox<float>(box, score, cls));
      }
      if (0 == bbox.size()) continue;
      // Apply NMS
      int n_boxes = bbox.size();
      // fyk: GPU nms
      if (caffe::Caffe::mode() == caffe::Caffe::GPU && FrcnnParam::test_use_gpu_nms) {
        int box_dim = 5;
        // sort score if use naive nms
        if (FrcnnParam::test_soft_nms == 0) {
          sort(bbox.begin(), bbox.end());
          box_dim = 4;
        }
        std::vector<float> boxes_host(n_boxes * box_dim);
        for (int i=0; i < n_boxes; i++) {
          for (int k=0; k < box_dim; k++)
            boxes_host[i * box_dim + k] = bbox[i][k];
        }
        int keep_out[n_boxes];//keeped index of boxes_host
        int num_out;//how many boxes are keeped
        // call gpu nms, currently only support naive nms
        //-----------NMS cascade increase--------------
        //_nms(&keep_out[0], &num_out, &boxes_host[0], n_boxes, box_dim, FrcnnParam::test_nms + 0.1 * stage);
        _nms(&keep_out[0], &num_out, &boxes_host[0], n_boxes, box_dim, FrcnnParam::test_nms);
        //if (FrcnnParam::test_soft_nms == 0) { // naive nms
        //  _nms(&keep_out[0], &num_out, &boxes_host[0], n_boxes, box_dim, FrcnnParam::test_nms);
        //} else {
        //  _soft_nms(&keep_out[0], &num_out, &boxes_host[0], n_boxes, box_dim, FrcnnParam::test_nms, FrcnnParam::test_soft_nms);
        //}
        for (int i=0; i < num_out; i++) {
          results[out_idx].push_back(bbox[keep_out[i]]);
        }
      } else { // cpu
        if (FrcnnParam::test_soft_nms == 0) { // naive nms
          sort(bbox.begin(), bbox.end());
          vector<bool> select(bbox.size(), true);
          for (int i = 0; i < bbox.size(); i++)
            if (select[i]) {
              //if (bbox[i].confidence < FrcnnParam::test_score_thresh) break;
              for (int j = i + 1; j < bbox.size(); j++) {
                if (select[j]) {
                  if (get_iou(bbox[i], bbox[j]) > FrcnnParam::test_nms) {
                    select[j] = false;
                  }
                }
              }
              results[out_idx].push_back(bbox[i]);
            }
        } else {
          // soft-nms
          float sigma = 0.5;
          float score_thresh = 0.001;
          int N = bbox.size();
          for (int cur_box_idx = 0; cur_box_idx < N; cur_box_idx++) {
            // find max score box
            float maxscore = bbox[cur_box_idx][4];
            int maxpos = cur_box_idx;
            for (int i = cur_box_idx + 1; i < N; i++) {
              if (maxscore < bbox[i][4]) {
                maxscore = bbox[i][4];
                maxpos = i;
              }
            }
            //swap
            for (int t=0; t<5;t++) {
              float tt = bbox[cur_box_idx][t];
              bbox[cur_box_idx][t] = bbox[maxpos][t];
              bbox[maxpos][t] = tt;
            }
            for (int i = cur_box_idx + 1; i < N; i++) {
              float iou = get_iou(bbox[i], bbox[cur_box_idx]);
              float weight = 1;
              if (1 == FrcnnParam::test_soft_nms) { // linear
                if (iou > FrcnnParam::test_nms) weight = 1 - iou;
              } else if (2 == FrcnnParam::test_soft_nms) { // gaussian
                weight = exp(- (iou * iou) / sigma);
              } else { // original NMS
                if (iou > FrcnnParam::test_nms) weight = 0;
              }
              bbox[i][4] *= weight;
              if (bbox[i][4] < score_thresh) {
                // discard the box by swapping with last box
                for (int t=0; t<5;t++) {
                  float tt = bbox[i][t];
                  bbox[i][t] = bbox[N-1][t];
                  bbox[N-1][t] = tt;
                }
                N -= 1;
                i -= 1;
              }
            }
          }
          for (int i=0; i < N; i++) {
            results[out_idx].push_back(bbox[i]);
          }
        } //nms type switch
      } //cpu
    } // for cls
  } // for out_idx
}

} // FRCNN_API
