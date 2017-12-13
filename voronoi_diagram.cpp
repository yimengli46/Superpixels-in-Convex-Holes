#include <iostream>
#include <fstream>
#include <string>
#include <deque>
#include <algorithm>
#include <stdlib.h>
// CGAL
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/algorithm.h>
#include <CGAL/Fuzzy_sphere.h>
#include <CGAL/Search_traits_2.h>
//openCV
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
//boost
#include <boost/graph/adjacency_list.hpp> // for customizable graphs
#include <boost/graph/directed_graph.hpp> // A subclass to provide reasonable arguments to adjacency_list for a typical directed graph
#include <boost/graph/undirected_graph.hpp>// A subclass to provide reasonable arguments to adjacency_list for a typical undirected graph
#include "convex_hull_segmentation.h"
using namespace std;
using namespace cv;

typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_2 Point_d;
typedef K::Segment_2 Segment_d;
typedef K::Vector_2 Vector_d;
typedef K::Line_2 Line_d;
typedef CGAL::Search_traits_2<K> Traits;
typedef CGAL::Fuzzy_sphere<Traits> Fuzzy_circle;
typedef CGAL::Kd_tree<Traits> Tree;

bool segment_comparator (Segment_d s1, Segment_d s2) 
{
	float s1_len = s1.squared_length();
	float s2_len = s2.squared_length();
	return s1_len > s2_len;
}

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

Point_d generateRandomPointAround(Point_d &current_point, float min_dis)
{
	//non-uniform, favours points closer to the inner ring, leads to denser packings
	double r1 = fRand(0.0, 1.0);
	double r2 = fRand(0.0, 1.0);
	//random radius between mindist and 2 * mindist
	double PI = 3.1415926;
	float radius = min_dis * (r1 + 1);
	//random angle
	double angle = 2 * PI * r2;
	//the new point is generated around the point (x, y)
	double newX = current_point.x() + radius * cos(angle);
	double newY = current_point.y() + radius * sin(angle);
	return Point_d(newX, newY);
}

void draw_voronoi_diagram(string result_name, Mat &img, vector<Point_d> &seg_seeds, vector<bool> &bool_seeds, bool bool_sites)
{
	Size size = img.size();
    Rect rect(0, 0, size.width, size.height);
    // Create an instance of Subdiv2D
    Subdiv2D subdiv(rect);
    // Create a vector of points.
    vector<Point2f> points;
    for(int i = 0; i < seg_seeds.size(); i++)
    {
    	if(bool_seeds[i])
    	{
    		Point2f seed(seg_seeds[i].x(), seg_seeds[i].y());
    		if(0 <= seed.x  && seed.x < size.width && 0 <= seed.y && seed.y < size.height)
    			points.push_back(seed);
    	}
    }     
    // Insert points into subdiv
    for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
    {
        subdiv.insert(*it);
    }
    // Allocate space for Voronoi Diagram
    Mat img_voronoi = img.clone();
    vector<vector<Point2f> > facets;
    vector<Point2f> centers;
    subdiv.getVoronoiFacetList(vector<int>(), facets, centers);
    vector<Point> ifacet;
    vector<vector<Point> > ifacets(1);
 
    for( size_t i = 0; i < facets.size(); i++ )
    {
        ifacet.resize(facets[i].size());
        for( size_t j = 0; j < facets[i].size(); j++ )
            ifacet[j] = facets[i][j];
 
        Scalar color;
        color[0] = rand() & 255;
        color[1] = rand() & 255;
        color[2] = rand() & 255;
        //fillConvexPoly(img_voronoi, ifacet, color, 8, 0);
 
        ifacets[0] = ifacet;
        polylines(img_voronoi, ifacets, true, Scalar(), 1, CV_AA, 0);
        if(bool_sites)
        	circle(img_voronoi, centers[i], 1, Scalar(), CV_FILLED, CV_AA, 0);
    }
    imwrite(result_name, img_voronoi);
}

int voronoi_diagram_main(string input_name, float input_eps)
{
	float eps = input_eps;
	float eps_sqrt = sqrt(eps);

	// read in line segment detection result provided by LSD
	string shape_detection_file = "lsd-result.txt";
	ifstream in;
	in.open(shape_detection_file.c_str());
	if(in.fail())
    {
		printf("Can't open shape_detection_file: %s\n", shape_detection_file.c_str());
    }

    vector<Segment_d> segs;
    vector<bool> segs_bool; // indicate whether the line segment is still there
    int N_seg = 0;

	while(true)
	{
		string line = "";
		getline(in, line);
		if(line == "")
		  break;
		//printf("%s\n", line.c_str());
		stringstream ss(line);
		double x1, y1, x2, y2;
		ss >> x1;
		ss >> y1;
		ss >> x2;
		ss >> y2;
		//printf("x1 = %lf, y1 = %lf, x2 = %lf, y2 = %lf\n", x1, y1, x2, y2);
		Point_d p1(x1, y1);
		Point_d p2(x2, y2);
		Segment_d seg(p1, p2);
		segs.push_back(seg);
		segs_bool.push_back(true);
		//tree.insert(seg);
		N_seg ++;
	}
	cout << "N_seg = " << N_seg << endl;
	sort(segs.begin(), segs.end(), segment_comparator);

	Mat img = imread(input_name.c_str(), 1);
    cv::Mat img1 = Mat::zeros( Size(560, 425), CV_8UC3 );
	cv::Mat img2 = Mat::zeros( Size(560, 425), CV_8UC3 );
	cv::Mat img3 = Mat::zeros( Size(560, 425), CV_8UC3 );// for segment seeds
	cv::Mat img4 = Mat::zeros( Size(560, 425), CV_8UC3 );//for segment seeds
	cv::Mat img5 = Mat::zeros( Size(560, 425), CV_8UC3 );//for removing
	cv::Mat img6 = Mat::zeros( Size(560, 425), CV_8UC3 );//for concurrence
	cv::Mat img7 = Mat::zeros( Size(560, 425), CV_8UC3 );//for concurrence
	cv::Mat img8 = Mat::zeros( Size(560, 425), CV_8UC3 );//final result

	// draw lines segments on image1
	for(int i = 0; i < segs.size(); i++)
	{
		Point start(segs[i].source().x(), segs[i].source().y());
		Point end  (segs[i].target().x(), segs[i].target().y());
		line(img1, start,end,Scalar(255, 255, 255), 1, 8);
	}
	cv::imwrite("vd_img1.png", img1);

    //typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS> Graph;
  	//Graph g(segs.size()); // Create a graph with 3 vertices.
 
  	// The graph behaves as a new user would expect if the vertex container type is vector. That is, vertices can be indexed with an unsigned int.
  	//boost::add_edge(0, 1, g);
  	//boost::add_edge(1, 2, g);
  	vector<vector<int> > adjacencyList(segs.size());
  	vector<bool> adjacency_bool(segs.size()); // indicate whether the line segment is adjacent to the others
  	for(int i = 0; i < adjacency_bool.size(); i++)
  	{
  		adjacency_bool[i] = false;
  	}

    for(int i = 0; i < segs.size() - 1; i++)
    {
    	for(int j = i + 1; j < segs.size(); j++)
    	{
    		float temp = CGAL::squared_distance(segs[i], segs[j]);
    		if(temp < eps)
    		{
    			printf("p = %d, q = %d, dis = %f\n", i, j, temp);
    			adjacencyList[i].push_back(j);
    			adjacency_bool[i] = true;
    			adjacency_bool[j] = true;
    		}
    	}
    }

    printf("\nThe Adjacency List-\n");
    // Printing Adjacency List
    /*
    for (int i = 1; i < adjacencyList.size(); ++i) {
        printf("adjacencyList[%d] ", i);
         
        list<int>::iterator itr = adjacencyList[i].begin();
         
        while (itr != adjacencyList[i].end()) {
            printf(" -> %d", (*itr));
            ++itr;
        }
        printf("\n");
    }
    */

    // draw lines segments on image1
	for(int i = 0; i < segs.size(); i++)
	{
		Point start(segs[i].source().x(), segs[i].source().y());
		Point end  (segs[i].target().x(), segs[i].target().y());
		if(adjacency_bool[i])
			line(img2, start, end,Scalar(0, 0, 255), 2, 8);
		else
			line(img2, start, end,Scalar(255, 255, 255), 2, 8);
	}
	cv::imwrite("vd_img2.png", img2);

	vector<Point_d> seg_seeds;
	vector<bool>    bool_seeds;
	/*************************** step1: find points on the two sides of the segment ***********************/
	for(int i = 0; i < segs.size(); i++)
	{
		Segment_d current_seg = segs[i];

		Point_d mid_point = CGAL::midpoint(current_seg.source(), current_seg.end());

		Vector_d v1(current_seg.source(), current_seg.target()); //vector follow direction of the segment
		Vector_d v2(current_seg.target(), current_seg.source());
		v1 = v1/sqrt(v1.squared_length());
		v2 = v2/sqrt(v2.squared_length());
		//cout << "v1.length = " << v1.squared_length() << endl;
		//cout << "v2.length = " << v2.squared_length() << endl;
		Vector_d v_clockwise = v1.perpendicular(CGAL::CLOCKWISE);
		Vector_d v_countercw = v1.perpendicular(CGAL::COUNTERCLOCKWISE);
	
		//cout << "x = " << v1_clockwise.x() << ", y = " << v1_clockwise.y() << endl;
		vector<Point_d> vec_points_on_segment;
		//bool bool_temp = CGAL::do_intersect(current_seg, mid_point);
		//cout << "midpoint on segment: " << bool_temp << endl;
		vec_points_on_segment.push_back(mid_point);
		int num = 1;
		while(true)
		{
			Point_d p1 = mid_point + num * 2 * eps_sqrt * v1;
			Point_d p2 = mid_point + num * 2 * eps_sqrt * v2;
			float temp = CGAL::squared_distance(p1, current_seg);
			if(temp < 0.25 * eps)
			{
				vec_points_on_segment.push_back(p1);
				vec_points_on_segment.push_back(p2);
			}
			else
				break;
			num ++;
		}
		//cout << "vec_points_on_segment.size = " << vec_points_on_segment.size() << endl;

		for(int j = 0; j < vec_points_on_segment.size(); j++)
		{
			Point_d current_point = vec_points_on_segment[j];
			Point_d p1 = current_point + eps_sqrt * v_clockwise;
			seg_seeds.push_back(p1);
			bool_seeds.push_back(true);
			Point_d p2 = current_point + eps_sqrt * v_countercw;
			seg_seeds.push_back(p2);
			bool_seeds.push_back(true);
		}
	}
	//remove closed seeds
	int count_bool = 0;
	for(int i = 0; i < bool_seeds.size(); i++)
		if(bool_seeds[i])
			count_bool ++;
	printf("Have %d seeds in the beginning ......\n", count_bool);
	for(int i = 0; i < seg_seeds.size() - 1; i++)
    {
    	for(int j = i + 1; j < seg_seeds.size(); j++)
    	{
    		if(bool_seeds[j])
    		{
	    		float temp = sqrt(CGAL::squared_distance(seg_seeds[i], seg_seeds[j]));
	    		if(temp < 2 * eps_sqrt)
	    		{
	    			//printf("p = %d, q = %d, dis = %f\n", i, j, temp);
	    			bool_seeds[j] = false;
	    		}
    		}
    	}
    }
    count_bool = 0;
    for(int i = 0; i < bool_seeds.size(); i++)
		if(bool_seeds[i])
			count_bool ++;
    printf("Have %d seeds in the end ..........\n", count_bool);
	// draw lines segments on image3
	for(int i = 0; i < segs.size(); i++)
	{
		Point start(segs[i].source().x(), segs[i].source().y());
		Point end  (segs[i].target().x(), segs[i].target().y());
		line(img3, start,end,Scalar(255, 255, 255), 1, 8);
		line(img4, start,end,Scalar(255, 255, 255), 1, 8);
	}
	for(int i = 0; i < seg_seeds.size(); i++)
	{
		if(bool_seeds[i])
		{
			Point seed(seg_seeds[i].x(), seg_seeds[i].y());
			circle(img3, seed, 1, Scalar(255, 255, 0), 1, 8);
			circle(img4, seed, 1, Scalar(255, 255, 0), 1, 8);
		}
		else
		{
			Point seed(seg_seeds[i].x(), seg_seeds[i].y());
			circle(img4, seed, 1, Scalar(0, 255, 255), 1, 8);
		}
	}

	cv::imwrite("vd_img3.png", img3);
	cv::imwrite("vd_img4.png", img4);
	/*************** step 2: find junction seeds *******************************/
	vector<Point_d> junction_seeds;
	vector<bool>    bool_junction_seeds;





	/************************* step3: homogenization, poisson-disk sampling ****************/
	vector<Point_d> homo_seeds;
	vector<bool>    bool_homo_seeds;

	//create the grid
	//float cellSize = sqrt(2 * eps) / sqrt(2);
	//Mat grid(Size(ceil(img.cols / cellSize), ceil(img.rows / cellSize)), CV_8U, Scalar::all(0));
	Tree tree;
	//cout << "grid.size = " << grid.size() << endl;

	deque<Point_d> processList;
	// homo_seeds is the samplePoints list

	//generate the first list, for me push in the seg_seeds and junction_seeds
	for(int i = 0; i < seg_seeds.size(); i++)
	{
		if(bool_seeds[i])
		{
			processList.push_back(seg_seeds[i]);
			//grid.at<uchar>((int)(seg_seeds[i].y() / cellSize), (int)(seg_seeds[i].x() / cellSize)) = 1;
			tree.insert(seg_seeds[i]);
		}
	}
	//random shuffle the queue
	random_shuffle (processList.begin(), processList.end());

	int newPoints_count = 30;
	//generate other points from points in queue
	while(!processList.empty())
	{
		Point_d current_point = processList.front();
		processList.pop_front();
		for(int i = 0; i < newPoints_count; i++)
		{
			Point_d new_point = generateRandomPointAround(current_point, 2 * eps_sqrt);
			if(0 <= new_point.x()  && new_point.x() < img.cols && 0 <= new_point.y() && new_point.y() < img.rows)
			{
				bool flag = true;
				/*
				if(grid.at<uchar>((int)(new_point.y() / cellSize), (int)(new_point.x() / cellSize)) > 0)
				{
					//cout << "grid is true" << endl;
					flag = false;
				}
				*/
				if(flag)
				{
					Fuzzy_circle default_range(new_point, 2 * eps_sqrt);
					list<Point_d> result;
					tree.search(back_inserter( result ), default_range);
					if(result.size() > 0)
					{
						//cout << "result.size > 0" << endl;
						flag = false;
					}
				}
				
				if(flag)
				{
					processList.push_back(new_point);
					homo_seeds.push_back(new_point);
					bool_homo_seeds.push_back(true);
					//grid.at<uchar>((int)(new_point.y() / cellSize), (int)(new_point.x() / cellSize)) = 1;
					tree.insert(new_point);
				}
			}
		}
		random_shuffle (processList.begin(), processList.end());
	}



	/*************** step 4: draw voronoi diagram ****************************/
	//Mat img_clone = img.clone();
	// Rectangle to be used with Subdiv2D
    draw_voronoi_diagram("vd_img10.png", img, seg_seeds, bool_seeds, true);
    cout << "homo_seeds.size() = " << homo_seeds.size() << endl;

  	vector<Point_d> all_seeds;
  	vector<bool> all_bools;
  	for(int i = 0; i < seg_seeds.size(); i++)
  	{
  		all_seeds.push_back(seg_seeds[i]);
  		all_bools.push_back(bool_seeds[i]);
  	}
  	for(int i = 0; i < homo_seeds.size(); i++)
  	{
  		all_seeds.push_back(homo_seeds[i]);
  		all_bools.push_back(bool_homo_seeds[i]);
  	}
  	draw_voronoi_diagram("vd_img11.png", img, all_seeds, all_bools, true);
  	draw_voronoi_diagram("vd_img12.png", img, all_seeds, all_bools, false);
}