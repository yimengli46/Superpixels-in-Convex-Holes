#ifndef CONVEX_HULL_SEGMENTATION_HEADER
#define CONVEX_HULL_SEGMENTATION_HEADER

#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/*
double * LineSegmentDetection( int * n_out,
                               double * img, int X, int Y,
                               double scale, double sigma_scale, double quant,
                               double ang_th, double log_eps, double density_th,
                               double union_ang_th, int union_use_NFA, double union_log_eps,
                               int n_bins, int need_to_union,
                               int ** reg_img, int * reg_x, int * reg_y,
                               double length_threshold, double dist_threshold );
                               */

int my_lsd_main(string);
int shape_detection_main(string);
int voronoi_diagram_main(string, float);
#endif