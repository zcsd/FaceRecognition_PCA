/*PCA Process for tranning data*/
#ifndef MY_PCA_H
#define MY_PCA_H

#include <iostream>
#include <vector>
#include <float.h>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

class MyPCA {

public:
	MyPCA(vector<string>& _facesPath);
	void init(vector<string>& _facesPath);
    void getImgSize(vector<string>& _facesPath);
    void mergeMatrix(vector<string>& _facesPath);
    void getAverageVector();
    void subtractMatrix();
    void getBestEigenVectors(Mat _covarMatrix);
    Mat getFacesMatrix();
	Mat getAverage();
	Mat getEigenvectors();
    ~MyPCA();

private:
    int imgSize = -1;//Dimension of features
    int imgRows = -1;//row# of image
    Mat allFacesMatrix;
    Mat avgVector;
    Mat subFacesMatrix;
    Mat eigenVector;
};

#endif
