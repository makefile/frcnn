#ifndef VIS_LOGGER_H_
#define VIS_LOGGER_H_

#include <map>

//#ifdef USE_VISUALDL
//#include "visualdl/logic/sdk.h"
//#endif

/**
  * external logger tools, such as VisualDL, Tensorboard, .etc
  * only support logging of scalar using VisualDL currently.
  */
class VisLogger {
  public:
    VisLogger(const std::string dir="./log", int sync_cycle=30, const std::string mode="train");
    ~VisLogger();
    void log_scalar(const std::string scalar_name, int step, float value);
  private:
    void *logger;
    std::map<std::string, void *> scalar_map;
    void* get_scalar_by_name(const std::string scalar_name);
};

// C style, global VisLogger object
extern "C" {
    void log_scalar(const std::string scalar_name, int step, float value);
}

#endif // VIS_LOGGER_H_
