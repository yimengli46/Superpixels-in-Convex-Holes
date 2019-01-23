This repository is an C++ implementation for paper '[Image partitioning into convex polygons](https://hal.inria.fr/hal-01140320/document)'.  

To test the code, make sure that you have OpenCV, CGAL and Boost properly installed on your machine.

To compile the code, I suggest to use CMake.  
A CMakeLists.txt file is provided.  
To compile the code, follow the steps below.
```
$ mkdir build && cd build  
$ cmake ..  
$ make -j4
```

If no errors show up during the compilation, you will see a executable file named convex_hull_segmentation.  
To run it, the command is:
```
$ ./convex_hull_segmentation input_image epsilon_value. 
```
The default value for epsilon parameter is 40.0.  

There is a test_img folder with 5 images for testing.  
To test, simply run this command
```
$ ./convex_hull_segmentation ../test_imgs/0001.png 40.0.  
```
You will see the result image and all the intermediate results show up in the build folder.  
