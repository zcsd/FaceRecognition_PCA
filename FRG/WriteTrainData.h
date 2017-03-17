#ifndef WRITE_TRAIN_DATA
#define WRITE_TRAIN_DATA

#include <string>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "MyPCA.h"

using namespace std;
using namespace cv;

class WriteTrainData {
public:
    WriteTrainData(MyPCA _trainPCA);
    void project(MyPCA _trainPCA);
    Mat getFacesInEigen();
    void writeTrainFacesData();
    ~WriteTrainData();

private:
    Mat trainFacesInEigen;
    int numberOfFaces;
};

#endif

