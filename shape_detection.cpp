#include <iostream>
#include <fstream>
#include <string>
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

bool segment_comparator1 (Segment_d s1, Segment_d s2) 
{
	float s1_len = s1.squared_length();
	float s2_len = s2.squared_length();
	return s1_len > s2_len;
}

bool collinear_comparator(pair<float, int> p1, pair<float, int> p2)
{
	return p1.first <= p2.first;
}

bool near_collinear (Segment_d s1, Segment_d s2)
{
	// find the closest two points on the two segments
	float min_length = 1000;
	float dis_ac = CGAL::squared_distance(s1.source(), s2.source());
	float dis_ad = CGAL::squared_distance(s1.source(), s2.target());
	float dis_bc = CGAL::squared_distance(s1.target(), s2.source());
	float dis_bd = CGAL::squared_distance(s1.target(), s2.target());
  	vector<pair<float, int> > vec_floats;
  	vec_floats.push_back(make_pair(dis_ac, 0));
  	vec_floats.push_back(make_pair(dis_ad, 1));
  	vec_floats.push_back(make_pair(dis_bc, 2));
  	vec_floats.push_back(make_pair(dis_bd, 3));
  	sort(vec_floats.begin(), vec_floats.end(), collinear_comparator);
  	
  	float inner_product = 0.0;
  	float cosine = 0.0;
  	if(vec_floats[0].second == 0)//ac is closest
  	{
  		Vector_d v1(s1.source(), s1.target());
  		Vector_d v2(s2.source(), s2.target());
  		inner_product = v1 * v2;
		cosine = inner_product / sqrt(v1.squared_length()) / sqrt(v2.squared_length());
  	}
  	else if (vec_floats[0].second == 1)//ad
  	{
  		Vector_d v1(s1.source(), s1.target());
  		Vector_d v2(s2.target(), s2.source());
  		inner_product = v1 * v2;
		cosine = inner_product / sqrt(v1.squared_length()) / sqrt(v2.squared_length());
  	}
  	else if (vec_floats[0].second == 2)//bc
  	{
  		Vector_d v1(s1.target(), s1.source());
  		Vector_d v2(s2.source(), s2.target());
  		inner_product = v1 * v2;
		cosine = inner_product / sqrt(v1.squared_length()) / sqrt(v2.squared_length());
  	}
  	else//bd
  	{
  		Vector_d v1(s1.target(), s1.source());
  		Vector_d v2(s2.target(), s2.source());
  		inner_product = v1 * v2;
		cosine = inner_product / sqrt(v1.squared_length()) / sqrt(v2.squared_length());
  	}
  	
	return cosine < -0.75;
  	/*
	
	*/
}

bool near_parallel(Segment_d s1, Segment_d s2)
{
	Vector_d v1 = s1.to_vector();
	Vector_d v2 = s2.to_vector();

	float inner_product = v1 * v2;
	// same direction
	if(inner_product < 0)
		inner_product = inner_product * -1;
	
	float cosine = inner_product / sqrt(v1.squared_length()) / sqrt(v2.squared_length());
	return cosine > 0.75;

}


int shape_detection_main(string input_name)
{
	float eps = 40;
	RNG rng(12345);
	// read in line segment detection result provided by LSD
	string lineSegment_detection_file = "lsd-result.txt";
	ifstream in;
	in.open(lineSegment_detection_file.c_str());
	if(in.fail())
    {
		printf("Can't open lineSegment_detection_file: %s\n", lineSegment_detection_file.c_str());
    }

    vector<Segment_d> segs;
    vector<bool> segs_bool; // indicate whether the line segment is still there
    int N_seg = 0;
    Tree tree;

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
	//cout << "N_seg = " << N_seg << endl;
	sort(segs.begin(), segs.end(), segment_comparator1);
	//std::cout << "sqdist(Segment_2(p,q), m) = " << CGAL::squared_distance(segs[0], segs[1]) << std::endl;

    Mat img = imread(input_name.c_str(), 1);
    cv::Mat img1 = Mat::zeros( Size(560, 425), CV_8UC3 );
	cv::Mat img2 = Mat::zeros( Size(560, 425), CV_8UC3 );
	cv::Mat img3 = Mat::zeros( Size(560, 425), CV_8UC3 );// for merging
	cv::Mat img4 = Mat::zeros( Size(560, 425), CV_8UC3 );//for merging
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
	cv::imwrite("img1.png", img1);

	img6 = img.clone();
	// draw lines segments on image1
	for(int i = 0; i < segs.size(); i++)
	{
		Point start(segs[i].source().x(), segs[i].source().y());
		Point end  (segs[i].target().x(), segs[i].target().y());
		line(img6, start,end,Scalar(255, 0, 0), 2, 8);
	}
	cv::imwrite("img6.png", img6);


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
    			//printf("p = %d, q = %d, dis = %f\n", i, j, temp);
    			adjacencyList[i].push_back(j);
    			adjacency_bool[i] = true;
    			adjacency_bool[j] = true;
    		}
    	}
    }

    //printf("\nThe Adjacency List-\n");
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
	cv::imwrite("img2.png", img2);

	for(int r = 0; r < 3; r++)
	{
		//cout << "round = " << r << "*********************" << endl;
		// merging
		// check if two adjacent segments are collinear and distance < 0.5 eps
		vector<pair<int, int> > merged_seg_pairs;
		for (int i = 1; i < adjacencyList.size(); ++i)
		{
			if(segs_bool[i] && adjacency_bool[i])
			{
				vector<int> adjacent_vertices = adjacencyList[i];
				if(adjacent_vertices.size() > 0)
				{
					for(int j = 0; j < adjacent_vertices.size(); j++)
					{
						int i2 = adjacent_vertices[j];
						if(segs_bool[i2])
						{
							float temp = CGAL::squared_distance(segs[i], segs[i2]);
							bool bool_collinear = near_collinear(segs[i], segs[i2]);
							//cout << CGAL::compare_slope(segs[i], segs[j]) << endl;
							//int bool_collinear = CGAL::compare_slope(segs[i], segs[i2]);
							//bool temp_bool = (temp < eps);
							//cout << "temp smaller than eps: " << temp_bool << endl;
							if (temp < 0.5 * eps && bool_collinear > 0)
							{
								//printf("merge i = %d, i2 = %d\n", i, i2);
								segs_bool[i2] = false;
								merged_seg_pairs.push_back(make_pair(i, i2));
								Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
								Point start(segs[i].source().x(), segs[i].source().y());
								Point end  (segs[i].target().x(), segs[i].target().y());
								line(img3, start, end,Scalar(255, 0, 0), 2, 8);
								Point start2(segs[i2].source().x(), segs[i2].source().y());
								Point end2  (segs[i2].target().x(), segs[i2].target().y());
								line(img3, start2, end2,Scalar(0, 0, 255), 2, 8);
							}
						}
					}
					
				}
			}
		}
		// actually merge segment pairs
		for(int i = 0; i < merged_seg_pairs.size(); i++)
		{
			int j1 = merged_seg_pairs[i].first;
			int j2 = merged_seg_pairs[i].second;
			float max_length = 0;
			Segment_d max_seg;
			Segment_d seg1(segs[j1].source(), segs[j2].target());
			if(seg1.squared_length() > max_length)
			{
				max_length = seg1.squared_length();
				max_seg = seg1;
			}
			Segment_d seg2(segs[j1].source(), segs[j2].source());
			if(seg2.squared_length() > max_length)
			{
				max_length = seg2.squared_length();
				max_seg = seg2;
			}
			Segment_d seg3(segs[j1].target(), segs[j2].source());
			if(seg3.squared_length() > max_length)
			{
				max_length = seg3.squared_length();
				max_seg = seg3;
			}
			Segment_d seg4(segs[j1].target(), segs[j2].target());
			if(seg4.squared_length() > max_length)
			{
				max_length = seg4.squared_length();
				max_seg = seg4;
			}

			segs[j1] = max_seg;

			Point start(segs[j1].source().x(), segs[j1].source().y());
			Point end  (segs[j1].target().x(), segs[j1].target().y());
			//Segment_d seg(start, end);
			line(img4, start, end,Scalar(0, 255, 0), 2, 8);
		}

		// removing step
		// check if two segments are parallel
		vector<pair<int, int> > removed_seg_pairs;
		for (int i = 1; i < adjacencyList.size(); ++i)
		{
			if(segs_bool[i] && adjacency_bool[i])
			{
				vector<int> adjacent_vertices = adjacencyList[i];
				if(adjacent_vertices.size() > 0)
				{
					for(int j = 0; j < adjacent_vertices.size(); j++)
					{
						int i2 = adjacent_vertices[j];
						if(segs_bool[i2])
						{
							//bool bool_parallel = CGAL::parallel(segs[i], segs[i2]);
							bool bool_parallel = near_parallel(segs[i], segs[i2]);
							if (bool_parallel)
							{
								//printf("parallel i = %d, i2 = %d\n", i, i2);
								segs_bool[i2] = false;
								removed_seg_pairs.push_back(make_pair(i, i2));
								Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
								Point start(segs[i].source().x(), segs[i].source().y());
								Point end  (segs[i].target().x(), segs[i].target().y());
								line(img5, start, end,Scalar(255, 255, 0), 2, 8);
								Point start2(segs[i2].source().x(), segs[i2].source().y());
								Point end2  (segs[i2].target().x(), segs[i2].target().y());
								line(img5, start2, end2,Scalar(0, 255, 255), 2, 8);
							}
						}
					}
					
				}
			}
		}

		// Concurrence

	}


	// draw lines segments on image8
	vector<bool> draw_bool;
	for(int i = 0; i < segs.size(); i++)
	{
		draw_bool.push_back(false);
	}
	for(int i = 0; i < segs.size(); i++)
	{
		if(segs_bool[i])
		{
			Point start(segs[i].source().x(), segs[i].source().y());
			Point end  (segs[i].target().x(), segs[i].target().y());
			if(adjacency_bool[i])
			{
				vector<int> adjacent_vertices = adjacencyList[i];
				for(int j = 0; j < adjacent_vertices.size(); j++)
					draw_bool[adjacent_vertices[j]] = true;
				if(draw_bool[i])
					line(img8, start, end,Scalar(255, 255, 0), 1, 8);
				else if(adjacent_vertices.size() > 0)
					line(img8, start, end,Scalar(0, 0, 255), 1, 8);
				else
					line(img8, start, end,Scalar(255, 0, 0), 1, 8);
			}
			else
				line(img8, start, end,Scalar(255, 255, 255), 1, 8);
		}
		
	}
	cv::imwrite("img8.png", img8);
	
	imwrite("img3.png", img3);
	imwrite("img4.png", img4);
	imwrite("img5.png", img5);

	int count_true = 0;
	for(int i = 0; i < segs_bool.size(); i++)
	{
		if(segs_bool[i])
			count_true ++;
	}
	cout << "number of survived segments = " << count_true << endl;

	//write result line segments to file
	string shape_detection_file = "02.shapeDetection.result";
	ofstream out;
	out.open(shape_detection_file.c_str());
	if(out.fail())
    {
		printf("Can't open shape_detection_file: %s\n", shape_detection_file.c_str());
    }
    for(int i = 0; i < segs_bool.size(); i++)
    {
    	if(segs_bool[i])
    	{
    		out << segs[i].source().x() << " ";
    		out << segs[i].source().y() << " ";
    		out << segs[i].target().x() << " ";
    		out << segs[i].target().y() << endl;
    	}
    }

    //draw segment anchors



	return 0;

}

