/*Write trainging data fo txt file*/
#ifndef WRITE_TRAIN_DATA
#define WRITE_TRAIN_DATA

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "MyPCA.h"

using namespace std;
using namespace cv;

class WriteTrainData {
public:
    WriteTrainData(MyPCA _trainPCA, vector<string>& _trainFacesID);
    void project(MyPCA _trainPCA);
    Mat getFacesInEigen();
    void writeTrainFacesData(vector<string>& _trainFacesID);
    void writeMean(Mat avg);
    void writeEigen(Mat eigen);
    ~WriteTrainData();

private:
    Mat trainFacesInEigen;
    int numberOfFaces;
};

#endif

