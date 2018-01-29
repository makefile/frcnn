#include <cstdlib>
#include <string>
#include "logger/vis_logger.hpp"

#ifdef USE_VISUALDL
#include "visualdl/logic/sdk.h"
namespace vs = visualdl;
namespace cp = visualdl::components;
#endif

//static VisLogger vislogger;
//void log_scalar(const std::string scalar_name, int step, float value) 

VisLogger::VisLogger(const std::string dir, int sync_cycle, const std::string mode) {
#ifdef USE_VISUALDL
    vs::LogWriter *logger = new vs::LogWriter(dir, sync_cycle);
    logger->SetMode(mode);
    this->logger = (void *)logger;
#endif // USE_VISUALDL
}
void VisLogger::log_scalar(const std::string scalar_name, int step, float value) {
#ifdef USE_VISUALDL
    cp::Scalar<float> *scalar = (cp::Scalar<float> *)get_scalar_by_name(scalar_name);
    scalar->AddRecord(step, value);
#endif // USE_VISUALDL
}

#ifdef USE_VISUALDL
void* VisLogger::get_scalar_by_name(const std::string scalar_name) {
    auto search = scalar_map.find(scalar_name);
    if (scalar_map.end() == search) {
        auto tablet = ((vs::LogWriter *)logger)->AddTablet(std::string("scalars/") + scalar_name);
        cp::Scalar<float> *scalar = new cp::Scalar<float>(tablet);
        scalar_map.insert(std::make_pair(scalar_name, scalar));
        return scalar;
    }
    return (void*)search->second;
}

#endif // USE_VISUALDL

VisLogger::~VisLogger() {
#ifdef USE_VISUALDL
    for (auto it=scalar_map.begin(); it!=scalar_map.end(); ++it) {
        delete (cp::Scalar<float> *)it->second;
    }
    delete (vs::LogWriter *)logger;
#endif // USE_VISUALDL
}

// C style, global VisLogger object
//extern "C" {
    VisLogger vislogger;

    void log_scalar(const std::string scalar_name, int step, float value) {
        return vislogger.log_scalar(scalar_name, step, value);
    }
//}

