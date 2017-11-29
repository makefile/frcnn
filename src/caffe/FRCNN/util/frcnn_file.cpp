#include "caffe/FRCNN/util/frcnn_utils.hpp"

namespace caffe {

namespace Frcnn {

// ==================== file system
// return the filenames of all files that have the specified extension
// in the specified directory and all subdirectories
namespace fs = ::boost::filesystem;
std::vector<string> get_file_list(const std::string& path, const string& ext) {
  fs::path fs_path(path);
  vector<string> file_list;

  if(!fs::exists(fs_path) || !fs::is_directory(fs_path))
    return file_list;

  fs::recursive_directory_iterator it(fs_path);
  fs::recursive_directory_iterator endit;

  while (it != endit) {
    if (fs::is_regular_file(*it) && it->path().extension() == ext)
      file_list.push_back(it->path().filename().string());
    ++it;
  }

  return file_list;
}

template <typename Dtype>
void print_vector(std::vector<Dtype> data) {
  for (int i = 0; i < data.size(); i++) {
    LOG(ERROR) << data[i];
  }
}
template void print_vector(std::vector<float> data);
template void print_vector(std::vector<double> data);

std::string anchor_to_string(std::vector<float> data) {
  CHECK_EQ( data.size() % 4 , 0 ) << "Anchors Size is wrong : " << data.size();    
  char buff[200];
  std::string ans;
  for (size_t index = 0; index * 4 < data.size(); index++) {
    snprintf(buff, sizeof(buff), "%.2f %.2f %.2f %.2f", data[index*4+0], data[index*4+1], data[index*4+2], data[index*4+3]);
    ans += std::string(buff) + "\n";
  }
  return ans;
}

std::string float_to_string(const std::vector<float> data) {
  char buff[200];
  std::string ans;
  for (size_t index = 0; index < data.size(); index++) {
    snprintf(buff, sizeof(buff), "%.2f", data[index]);
    if( index == 0 ) ans = std::string(buff);
    else ans += ", " + std::string(buff);
  }
  return ans;
}

std::string float_to_string(const float *data) {
  const int n = sizeof(data) / sizeof(data[0]);
  return float_to_string( std::vector<float>(data, data+n) );
}

} // namespace frcnn

} // namespace caffe
