#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>

#include "caffe/FRCNN/util/frcnn_utils.hpp"

namespace caffe {

namespace Frcnn {

std::vector<std::string> &split(const std::string &s, char delim,
    std::vector<std::string> &elems) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, elems);
  return elems;
}

str_map parse_json_config(const std::string file_path) {
  std::ifstream ifs(file_path.c_str());
  std::map<std::string, std::string> json_map;
  boost::property_tree::ptree pt;
  boost::property_tree::read_json(ifs, pt);

  boost::property_tree::basic_ptree<std::string, std::string>::const_iterator
      iter = pt.begin();

  for (; iter != pt.end(); ++iter) {
    json_map[iter->first.data()] = iter->second.data();
  }
  return json_map;
}

std::string extract_string(std::string target_key, 
    str_map& default_map) {
  std::string target_str;
  if (default_map.count(target_key) > 0) {
    target_str = default_map[target_key];
  } else {
    LOG(FATAL) << "Can not find target_key : " << target_key;
  }
  return target_str;
}

float extract_float(std::string target_key, 
    str_map& default_map) {
  std::string target_str = extract_string(target_key, default_map);
  return atof(target_str.c_str());
}

int extract_int(std::string target_key, 
    str_map& default_map) {
  std::string target_str = extract_string(target_key, default_map);
  return atoi(target_str.c_str());
}

std::vector<float> extract_vector(std::string target_key,
     str_map& default_map) {
  std::string target_str = extract_string(target_key, default_map);
  std::vector<float> results;
  std::vector<std::string> elems = split(target_str, ',');

  for (std::vector<std::string>::const_iterator it = elems.begin();
       it != elems.end(); ++it) {
    results.push_back(atof((*it).c_str()));
  }
  return results;
}

} // namespace frcnn

} // namespace caffe
