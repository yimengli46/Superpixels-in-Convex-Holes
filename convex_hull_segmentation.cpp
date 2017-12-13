#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "convex_hull_segmentation.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	if(argc != 2 && argc != 3)
		cout << "usage: ./convex_hull_segmentation path_of_input_image epsillon_value(default = 40.0)" << endl;
	else
	{
		string img_name = argv[1];
		Mat img = imread(img_name, 1);
		if(!img.data)
		{
			cout << "Cannot open input image." << endl;
			return 0;
		}
		float eps = 0.0;
		if(argc == 3)
		{
			string fs(argv[2]);
			stringstream ss;
			ss << fs;
			ss >> fs;
		}
		else
			eps = 40.0;
		my_lsd_main(img_name);
		shape_detection_main(img_name);
		voronoi_diagram_main(img_name, eps);
	}
	return 0;
}