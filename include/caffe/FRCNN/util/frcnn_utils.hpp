// ------------------------------------------------------------------
// Xuanyi . Refer to Dong Jian
// 2016/03/29
// ------------------------------------------------------------------
#ifndef CAFFE_FRCNN_UTILS_HPP_
#define CAFFE_FRCNN_UTILS_HPP_

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <exception>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <map>
#include <queue>


#include "opencv2/core/version.hpp"

#if CV_MAJOR_VERSION == 2
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

#elif CV_MAJOR_VERSION == 3
#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#endif

#include <boost/algorithm/string.hpp>
#include "boost/filesystem.hpp"

#include <glog/logging.h>

#include "caffe/common.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"

//fyk for use math M_PI, fabs etc
#define _USE_MATH_DEFINES 
#include <cmath>
/*
#include <cstdlib>
template <typename Dtype>
inline Dtype mabs(Dtype x) { // Dtype in Caffe only has float and double
  if (typeid(Dtype) == typeid(double)) {
      return fabs(x);
  } else if (typeid(Dtype) == typeid(float)) {
      return fabsf(x);
  } else if (typeid(Dtype) == typeid(int)) {
      return abs(x);
  }
}
*/

namespace caffe {

namespace Frcnn {

class DataPrepare {
public:
  DataPrepare() {
    rois.clear();
    ok = false;
  }
  inline string GetImagePath(string root = "") {
    CHECK(this->ok) << "illegal status(ok=" << ok << ")";
    return root + image_path;
  }
  inline int GetImageIndex() {
    CHECK(this->ok) << "illegal status(ok=" << ok << ")";
    return image_index;
  }
  inline vector<vector<float> > GetRois(bool include_diff = false) {
    CHECK(this->ok) << "illegal status(ok=" << ok << ")";
    CHECK_EQ(this->rois.size(), this->diff.size());
    vector<vector<float> > _rois;
    for (size_t index = 0; index < this->rois.size(); index++) {
      if (include_diff == false && this->diff[index] == 1) continue;
      _rois.push_back( this->rois[index] );
    }
    return _rois;
  }
  inline bool load_WithDiff(std::ifstream &infile) {
    string hashtag;
    if(!(infile >> hashtag)) return ok=false;
    CHECK_EQ(hashtag, "#");
    CHECK(infile >> this->image_index >> this->image_path);
    int num_roi;
    CHECK(infile >> num_roi); 
    rois.clear(); diff.clear();
    for (int index = 0; index < num_roi; index++) {
      //int label, x1, y1, x2, y2; 
      int label;
      float cx, cy, w, h, theta; 
      int diff_;
      CHECK(infile >> label >> cx >> cy >> w >> h >> theta >> diff_) << "illegal line of " << image_path;//fyk add output info 
      //x1 --; y1 --; x2 --; y2 --;
      // CHECK LABEL
      CHECK(label>0 && label<FrcnnParam::n_classes) << "illegal label : " << label << ", should >= 1 and < " << FrcnnParam::n_classes;
      //CHECK_GE(x2, x1) << "illegal coordinate : " << x1 << ", " << x2 << " : " << this->image_path; 
      //CHECK_GE(y2, y1) << "illegal coordinate : " << y1 << ", " << y2 << " : " << this->image_path;
      CHECK(cx >= 0 && cy >= 0 && w > 0 && h > 0 ) << "illegal box property (cx,cy,w,h):" << cx << ", " << cy << ", " << w << ", " << h << ", " << this->image_path;
      CHECK_GT(theta, -M_PI_2) << "illegal box property: " << theta << " should in (-π/2,π/2] : " << this->image_path;
      CHECK_LE(theta, M_PI_2) << "illegal box property: " << theta << " should in (-π/2,π/2] : " << this->image_path;
      vector<float> roi(DataPrepare::NUM);
      roi[DataPrepare::LABEL] = label;
      roi[DataPrepare::CX] = cx;
      roi[DataPrepare::CY] = cy;
      roi[DataPrepare::W] = w;
      roi[DataPrepare::H] = h;
      roi[DataPrepare::THETA] = theta;
      rois.push_back(roi);
      diff.push_back(diff_);
    }
    return ok=true;
  }
  //enum RoiDataField { LABEL, X1, Y1, X2, Y2, NUM };
  //NUM is only used for counting size of box location properties.
  enum RoiDataField { LABEL, CX, CY, W, H, THETA, NUM };

private:
  vector<vector<float> > rois;
  vector<int> diff;
  string image_path;
  int image_index;
  bool ok;
};

// image and box
template <typename Dtype>
class Point4f {
public:
  Dtype Point[4]; // x1 y1 x2 y2
  Point4f(Dtype x1 = 0, Dtype y1 = 0, Dtype x2 = 0, Dtype y2 = 0) {
    Point[0] = x1; Point[1] = y1;
    Point[2] = x2; Point[3] = y2;
  }
  Point4f(const float data[4]) {
    for (int i=0;i<4;i++) Point[i] = data[i]; 
  }
  Point4f(const double data[4]) {
    for (int i=0;i<4;i++) Point[i] = data[i]; 
  }
  Point4f(const Point4f &other) { memcpy(Point, other.Point, sizeof(Point)); }
  Dtype& operator[](const unsigned int id) { return Point[id]; }
  const Dtype& operator[](const unsigned int id) const { return Point[id]; }

  string to_string() const {
    char buff[100];
    snprintf(buff, sizeof(buff), "%.1f %.1f %.1f %.1f", Point[0], Point[1], Point[2], Point[3]);
    return string(buff);
  }

};
//fyk add
template <typename Dtype>
class Point5f {
public:
  Dtype Point[5]; // cx1, cy1, w, h, theta
  Point5f(Dtype cx1 = 0, Dtype cy1 = 0, Dtype w = 0, Dtype h = 0, Dtype theta = 0) {
    Point[0] = cx1; Point[1] = cy1;
    Point[2] = w; Point[3] = h;
    Point[4] = theta;
  }
  Point5f(const float data[5]) {
    for (int i=0;i<5;i++) Point[i] = data[i]; 
  }
  Point5f(const double data[5]) {
    for (int i=0;i<5;i++) Point[i] = data[i]; 
  }
  Point5f(const Point5f &other) { memcpy(Point, other.Point, sizeof(Point)); }
  Dtype& operator[](const unsigned int id) { return Point[id]; }
  const Dtype& operator[](const unsigned int id) const { return Point[id]; }

  string to_string() const {
    char buff[100];
    snprintf(buff, sizeof(buff), "%.1f %.1f %.1f %.1f %.1f", Point[0], Point[1], Point[2], Point[3], Point[4]);
    return string(buff);
  }

};

template <typename Dtype>
class BBox : public Point4f<Dtype> {
public:
  Dtype confidence;
  int id;

  BBox(Dtype x1 = 0, Dtype y1 = 0, Dtype x2 = 0, Dtype y2 = 0,
       Dtype confidence = 0, int id = 0)
      : Point4f<Dtype>(x1, y1, x2, y2), confidence(confidence), id(id) {}
  BBox(Point4f<Dtype> box, Dtype confidence_ = 0, int id = 0)
      : Point4f<Dtype>(box), confidence(confidence_), id(id) {}

  BBox &operator=(const BBox &other) {
    memcpy(this->Point, other.Point, sizeof(this->Point));
    confidence = other.confidence;
    id = other.id;
    return *this;
  }

  bool operator<(const class BBox &other) const {
    if (confidence != other.confidence)
      return confidence > other.confidence;
    else
      return id < other.id;
  }

  inline string to_string() const {
    char buff[100];
    snprintf(buff, sizeof(buff), "cls:%3d -- (%.3f): %.2f %.2f %.2f %.2f", id,
             confidence, this->Point[0], this->Point[1], this->Point[2], this->Point[3]);
    return string(buff);
  }

  inline string to_short_string() const {
    char buff[100];
    snprintf(buff, sizeof(buff), "cls:%1d -- (%.2f)", id, confidence);
    return string(buff);
  }
};
//fyk
template <typename Dtype>
class RBBox : public Point5f<Dtype> {
public:
  Dtype confidence;
  int id;

  RBBox(Dtype cx = 0, Dtype cy = 0, Dtype w = 0, Dtype h = 0, Dtype theta = 0,
       Dtype confidence = 0, int id = 0)
      : Point5f<Dtype>(cx, cy, w, h, theta), confidence(confidence), id(id) {}
  RBBox(Point5f<Dtype> box, Dtype confidence_ = 0, int id = 0)
      : Point5f<Dtype>(box), confidence(confidence_), id(id) {}

  RBBox &operator=(const RBBox &other) {
    memcpy(this->Point, other.Point, sizeof(this->Point));
    confidence = other.confidence;
    id = other.id;
    return *this;
  }

  bool operator<(const class RBBox &other) const {
    if (confidence != other.confidence)
      return confidence > other.confidence;
    else
      return id < other.id;
  }

  inline string to_string() const {
    char buff[100];
    snprintf(buff, sizeof(buff), "cls:%3d -- (%.3f): %.2f %.2f %.2f %.2f %.2f", id,
             confidence, this->Point[0], this->Point[1], this->Point[2], this->Point[3], this->Point[4]);
    return string(buff);
  }

  inline string to_short_string() const {
    char buff[100];
    snprintf(buff, sizeof(buff), "cls:%1d -- (%.2f)", id, confidence);
    return string(buff);
  }
};

template <typename Dtype>
class TrackLet : public BBox<Dtype> {
public:
  int tracklet;
  TrackLet(Dtype x1 = 0, Dtype y1 = 0, Dtype x2 = 0, Dtype y2 = 0, Dtype confidence = 0, int id = 0, int _tracklet = 0):
    BBox<Dtype>(x1, y1, x2, y2, confidence, id), tracklet(_tracklet){};
  TrackLet(BBox<Dtype> box, int _tracklet = 0):
    BBox<Dtype>(box), tracklet(_tracklet){};
  TrackLet(Point4f<Dtype> box, Dtype confidence = 0, int id = 0, int _tracklet = 0):
    BBox<Dtype>(box, confidence, id), tracklet(_tracklet){};
  inline string to_string() const {
    char buff[100];
    snprintf(buff, sizeof(buff), "cls:%3d,let:%3d -- (%.3f): %.2f %.2f %.2f %.2f", this->id, this->tracklet,
             this->confidence, this->Point[0], this->Point[1], this->Point[2], this->Point[3]);
    return string(buff);
  }
};
//fyk not used by now,so i didn't adpat it to deal with angle
template <typename Dtype>
class VidPrepare {
public:
  VidPrepare() {
    ok = false;
    prefetch_rng_.reset();
    this->current_index = -1;
  }
  inline void init(const unsigned int seed = 0) {
    _image_dataset.clear();
    _objects.clear();
    prefetch_rng_.reset(new Caffe::RNG(seed));
  }
  inline bool load_data(std::ifstream &infile) {
    if(!(infile >> HASH)) return ok=false;
    CHECK_EQ(HASH, "#");
    CHECK(infile >> this->folder);
    CHECK(infile >> this->num_image >> this->height >> this->width);
    int x1, y1, x2, y2;
    int track_let, label;
    for (int index = 0; index < this->num_image; index++ ) {
      string image; int num_rois;
      CHECK(infile >> image >> num_rois);
      _image_dataset.push_back(image);
      vector<TrackLet<Dtype> > objects;

      for (int roi_ = 0; roi_ < num_rois; roi_++ ) {
        CHECK(infile >> track_let >> label >> x1 >> y1 >> x2 >> y2);
        TrackLet<Dtype> cobject(x1, y1, x2, y2, 1, label, track_let);
        CHECK(label>0 && label<FrcnnParam::n_classes) << "illegal label : " << label << ", should >= 1 and < " << FrcnnParam::n_classes;
        CHECK_GE(x1, 0) << cobject.to_string();
        CHECK_GE(y1, 0) << cobject.to_string();
        CHECK_LT(x1, this->width) << "Width : " << this->width << cobject.to_string();
        CHECK_LT(y1, this->height) << "Height : " << this->height << cobject.to_string();
        objects.push_back(cobject);
      }

      _objects.push_back(objects);
    }
    CHECK_EQ(_image_dataset.size(), _objects.size());
    return ok = true;
  }

  inline pair<vector<vector<float> >, string> Next() {
    CHECK(ok) << "Status is false";
    this->current_index = PrefetchRand() % _image_dataset.size();
    string image = folder + "/" + _image_dataset[current_index];
    const vector<TrackLet<Dtype> > &objects = _objects[current_index];
    vector<vector<float> > rois;
    for (size_t ii = 0; ii < objects.size(); ii++ ) {
      vector<float> roi(NUM);
      roi[LABEL] = objects[ii].id;
      roi[X1] = objects[ii][0];
      roi[Y1] = objects[ii][1];
      roi[X2] = objects[ii][2];
      roi[Y2] = objects[ii][3];
      rois.push_back(roi);
    } 
    CHECK_EQ(rois.size(), objects.size());
    return make_pair(rois, image);
  } 
  
  inline string message() {
    CHECK(ok) << "Status is false";
    CHECK_GE(this->current_index, 0);
    CHECK_LT(this->current_index, int(_image_dataset.size()));
    char buff[100];
    snprintf(buff, sizeof(buff), "height : %d, width : %d " , this->height, this->width);
    return string(buff);
  }

  inline map<int,int> count_label() {
    CHECK(ok) << "Status is false";
    map<int, int> label_hist;
    for (size_t index = 0; index < _objects.size(); index++ ) {
      for (size_t oid = 0; oid < _objects[index].size(); oid++ ) {
        int label = _objects[index][oid].id;
        label_hist.insert(std::make_pair(label, 0));
        label_hist[label]++;
      }
    }
    return label_hist;
  }

  inline int H() {
    CHECK(ok) << "Status is false";
    return this->height;
  }

  inline int W() {
    CHECK(ok) << "Status is false";
    return this->width;
  }

  enum RoiDataField { LABEL, X1, Y1, X2, Y2, NUM };
private:
  string HASH;
  string folder;
  int num_image;
  int height;
  int width;
  int current_index;
  vector<string> _image_dataset;
  vector<vector<TrackLet<Dtype> > > _objects;
  bool ok;

  // Random Seed 
  shared_ptr<Caffe::RNG> prefetch_rng_;
  inline unsigned int PrefetchRand() {
    CHECK(prefetch_rng_);
    caffe::rng_t *prefetch_rng =
        static_cast<caffe::rng_t *>(prefetch_rng_->generator());
    return (*prefetch_rng)();
  }
};

template <typename Dtype>
Dtype get_iou(const Point4f<Dtype> &A, const Point4f<Dtype> &B);

template <typename Dtype>
vector<vector<Dtype> > get_ious(const vector<Point4f<Dtype> > &A, const vector<Point4f<Dtype> > &B);
template <typename Dtype>
vector<vector<Dtype> > skew_ious(const vector<Point5f<Dtype> > &A, const vector<Point5f<Dtype> > &B);

template <typename Dtype>
vector<Dtype> get_ious(const Point4f<Dtype> &A, const vector<Point4f<Dtype> > &B);
template <typename Dtype>
vector<Dtype> skew_ious(const Point5f<Dtype> &A, const vector<Point5f<Dtype> > &B);
template <typename Dtype>
Dtype skew_iou(const Point5f<Dtype> &A, const Point5f<Dtype> &B);
template <typename Dtype>
vector<vector<Dtype> > angle_diff(const vector<Point5f<Dtype> > &A, const vector<Point5f<Dtype> > &B) ;
template <typename Dtype>
Point4f<Dtype> rotate_outer_box_coordinates(const Point5f<Dtype> &rbox);

float get_scale_factor(int width, int height, int short_size, int max_long_size);

// config
typedef std::map<string, string> str_map;

str_map parse_json_config(const string file_path);

string extract_string(string target_key, str_map& default_map);

float extract_float(string target_key,  str_map& default_map);

int extract_int(string target_key, str_map& default_map);

vector<float> extract_vector(string target_key, str_map& default_map);

// file 
vector<string> get_file_list (const string& path, const string& ext);

template <typename Dtype>
void print_vector(vector<Dtype> data); 

string anchor_to_string(vector<float> data);

string float_to_string(const vector<float> data);

string float_to_string(const float *data);

} // namespace Frcnn

} // namespace caffe

#endif // CAFFE_FRCNN_UTILS_HPP_
