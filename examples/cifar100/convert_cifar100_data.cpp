//
// This script converts the CIFAR dataset to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_cifar_data input_folder output_db_file
// The CIFAR dataset could be downloaded at
//    http://www.cs.toronto.edu/~kriz/cifar.html

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "boost/scoped_ptr.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"

using caffe::Datum;
using boost::scoped_ptr;
using std::string;
namespace db = caffe::db;

const int kCIFARSize = 32;
const int kCIFARImageNBytes = 3072;
const int kCIFARTrainSize = 50000;
const int kCIFARTestSize = 10000;

void read_image(std::ifstream* file, int* coarse_label, int* fine_label, char* buffer) {
  char label_char;
  file->read(&label_char, 1);
  *coarse_label = label_char;
  file->read(&label_char, 1);
  *fine_label = label_char;
  file->read(buffer, kCIFARImageNBytes);
  return;
}

void convert_dataset(const string& input_folder, const string& output_folder,
    const string& db_type) {
  scoped_ptr<db::DB> train_db(db::GetDB(db_type));
  train_db->Open(output_folder + "/cifar100_train_" + db_type, db::NEW);
  scoped_ptr<db::Transaction> txn(train_db->NewTransaction());
  // Data buffer
  int coarse_label;
  int fine_label;
  char str_buffer[kCIFARImageNBytes];
  Datum datum;
  datum.set_channels(3);
  datum.set_height(kCIFARSize);
  datum.set_width(kCIFARSize);

  LOG(INFO) << "Writing Training data";
  // Open files
  LOG(INFO) << "Training Batch Size : " << kCIFARTrainSize;
  string batchFileName = input_folder + "/train.bin";
  std::ifstream train_file(batchFileName.c_str(), std::ios::in | std::ios::binary);
  CHECK(train_file) << "Unable to open train file # " << batchFileName;
  for (int itemid = 0; itemid < kCIFARTrainSize; ++itemid) {
    read_image(&train_file, &coarse_label, &fine_label, str_buffer);
    datum.set_label(fine_label);
    datum.set_data(str_buffer, kCIFARImageNBytes);
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(caffe::format_int(itemid, 5), out);
  }
  txn->Commit();
  train_db->Close();

  LOG(INFO) << "Writing Testing data";
  scoped_ptr<db::DB> test_db(db::GetDB(db_type));
  test_db->Open(output_folder + "/cifar100_test_" + db_type, db::NEW);
  txn.reset(test_db->NewTransaction());
  // Open files
  std::ifstream test_file((input_folder + "/test.bin").c_str(), std::ios::in | std::ios::binary);
  CHECK(test_file) << "Unable to open test file # " << (input_folder + "/test.bin");
  LOG(INFO) << "Testing Batch Size : " << kCIFARTestSize;
  for (int itemid = 0; itemid < kCIFARTestSize; ++itemid) {
    read_image(&test_file, &coarse_label, &fine_label, str_buffer);
    datum.set_label(fine_label);
    datum.set_data(str_buffer, kCIFARImageNBytes);
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(caffe::format_int(itemid, 5), out);
  }
  txn->Commit();
  test_db->Close();
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = 1;

  if (argc != 4) {
    printf("This script converts the CIFAR dataset to the leveldb format used\n"
           "by caffe to perform classification.\n"
           "Usage:\n"
           "    convert_cifar_data input_folder output_folder db_type\n"
           "Where the input folder should contain the binary batch files.\n"
           "The CIFAR dataset could be downloaded at\n"
           "    http://www.cs.toronto.edu/~kriz/cifar.html\n"
           "You should gunzip them after downloading.\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(string(argv[1]), string(argv[2]), string(argv[3]));
  }
  return 0;
}
