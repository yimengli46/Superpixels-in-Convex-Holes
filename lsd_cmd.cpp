#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include "lsd.h"
#include "convex_hull_segmentation.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <fstream>


using namespace cv;
using namespace std;


/*----------------------------------------------------------------------------*/
/*                                    Main                                    */
/*----------------------------------------------------------------------------*/
/** Main function call
 */
int my_lsd_main(string img_name)
{

  FILE * output;
  double * image;
  int X,Y;
  double * segs;
  int n;
  int dim = 7;
  //int * region;
  int regX,regY;
  int i,j;

  Mat img = imread(img_name.c_str(), 1);
  cvtColor(img, img, CV_BGR2GRAY);
  X = img.cols;
  Y = img.rows;
  img.convertTo(img, CV_64F, 1, 0);
  image = (double *) calloc( (size_t) (X*Y), sizeof(double) );
  double *ptrDst[Y];
  for(int j = 0; j < Y; j++)
  {
  	ptrDst[j] = img.ptr<double>(j);
  	for(i = 0; i < X; i++)
  	{
  		image[i + j * X] = ptrDst[j][i];
  	}
  }


  //cout << "X = " << X << "Y = " << Y << endl;
  /* execute LSD */
  segs = LineSegmentDetection(&n, image, X, Y, 0.50, 0.6, 2.0, 22.50, 0.0, 0.70, 7.0, 0, -1.0, 1024, 0, NULL, &regX, &regY, 0.0, 5.0);

  output = fopen("lsd-result.txt", "w");
  //cout << "n = " << n << endl;
  //if( output == NULL ) error("Error: unable to open ASCII output file.");
  for(i=0;i<n;i++)
    {
      for(j=0;j<dim;j++)
        fprintf(output,"%f ",segs[i*dim+j]);
      fprintf(output,"\n");
    }

  /* free memory */
  free( (void *) image );
  free( (void *) segs );
  //free_arguments(arg);
  return 0;
}
/*----------------------------------------------------------------------------*/
