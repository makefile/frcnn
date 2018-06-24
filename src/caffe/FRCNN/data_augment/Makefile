all:
	g++ -std=c++11 `pkg-config opencv --cflags` *.cpp \
	-o test `pkg-config opencv --libs` \
	#-I /usr/local/opencv3/include \
	#-L /usr/local/opencv3/lib \
	#-lopencv_core -lopencv_highgui -lopencv_imgproc \
	#-pthread
	#-lopencv_imgcodecs
