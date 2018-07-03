#include "api/util/blowfish.hpp"
#include <cstdio>
#include <vector>
#include <cstring>

void show_usage(char* name) {
    printf("Encrypt/Decrypt tool.\n"
        "Usage: %s [enc|dec] <KEY> <FILE> <OUTPUT FILE>\n"
        "      enc - encrypt the file\n"
        "      dec - decrypt the file\n", name);
}

int main(int argc, char** argv) {
  //LOG(FATAL) << "Deprecated. Use caffe device_query "
  //              "[--device_id=0] instead.";
  if (argc < 5) {
    show_usage(argv[0]);
    //exit(0);
    return 0;
  }

  //std::string key(argv[2]);
  //std::vector<char> v_key(key.begin(), key.end());
  std::vector<char> v_key(argv[2], argv[2]+strlen(argv[2]));
  Blowfish bf(v_key);

  if (strncmp("enc", argv[1], 3)==0) {
    bf.Encrypt(argv[3], argv[4]);
  } else if (strncmp("dec", argv[1], 3)==0) {
    bf.Decrypt(argv[3], argv[4]);
  } else {
    show_usage(argv[0]);
  }
  return 0;
}
