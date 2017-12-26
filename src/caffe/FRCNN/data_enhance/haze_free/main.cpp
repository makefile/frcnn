
#include "haze.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <thread>
//g++ -std=c++11 

using namespace std;

std::thread::id main_thread_id = std::this_thread::get_id();
char *img_dir, *out_img_dir;
int N_thread = 8;
int N_img_per_thread = 10000/N_thread;
void haze_thread(int start_idx) {
	std::cout << "starting from " << start_idx << std::endl;
	char img_name[256],out_img_name[256];
	for (int i=0;i<N_img_per_thread;i++) {
		int name_idx = start_idx + i;
		if(i % 100 == 0) std::cout << name_idx << std::endl;
		sprintf(img_name,"%s/%06d.jpg",img_dir,name_idx);
		sprintf(out_img_name,"%s/%06d.jpg",out_img_dir,name_idx);
		remove_haze(img_name, out_img_name);
	}
}
int main_test(int argc, char* argv[]) {
	img_dir = argv[1];
	out_img_dir = argv[2];
	if(argc>3) {
		N_thread = atoi(argv[3]);
		N_img_per_thread = 10000/N_thread;
	}

	std::thread threads[N_thread];                         // 默认构造线程

	std::cout << "Spawning threads...\n";
	int start_idx = 5001;
	for (int i = 0; i < N_thread; ++i)
		threads[i] = std::thread(haze_thread, start_idx + i * N_img_per_thread);   // move-assign threads
	std::cout << "Done spawning threads. Now waiting for them to join:\n";
	for (auto &thread : threads)
		thread.join();
	std::cout << "All threads joined!\n";
	return 0;
}

