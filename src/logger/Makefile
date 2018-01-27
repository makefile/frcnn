CXX_SRCS := $(shell find . -name "*.cpp")
CXX_OBJS := ${CXX_SRCS:.cpp=.o}
CXXFLAGS += -O3 -std=c++11 -fPIC
VISUALDL_LIB := /home/s04/fyk/VisualDL/core.so
OUT_DIR := ../../distribute/lib
OUT_SO := $(OUT_DIR)/libvis_logger.so

$(OUT_SO): $(CXX_OBJS)
	@ echo CXX/LD -o $@ $<
	@ mkdir -p ../../distribute/lib/
	@ g++ -shared -o $@ $^ $(CXXFLAGS) \
    -l:$(VISUALDL_LIB) #-L/home/s04/fyk/protobuf/lib -lprotobuf
	@ rm *.o
	cp $(VISUALDL_LIB) $(OUT_DIR)
	@ echo done
%.o: %.cpp
	@ echo CXX/LD -o $@ $<
	@ g++ -c $< \
    $(CXXFLAGS) \
    -DUSE_VISUALDL \
    -I ../../include \
    -I /home/s04/fyk/VisualDL \
    -I /home/s04/fyk/protobuf-3.1.0/include

clean:
	rm *.o $(OUT_SO)
