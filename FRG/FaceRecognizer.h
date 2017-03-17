#ifndef FACE_RECOGNIZER_H
#define FACE_RECOGNIZER_H

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

class FaceRecognizer {
public:
    FaceRecognizer(Mat _testImg, Mat _avgVec, Mat _eigenVec, Mat _facesInEigen, vector<int>& _trainFacesID);
    void prepareFace(Mat _testImg);
    void projectFace(Mat testVec, Mat _avgVec, Mat _eigenVec);
    void recognize(Mat testPrjFace, Mat _facesInEigen, vector<int>& _trainFacesID);
    int getClosetFaceID();
    double getClosetDist();
    ~FaceRecognizer();

private:
    Mat testVec;
    Mat testPrjFace;
    int closetFaceID = -1;
    double closetFaceDist = 4000;
};

#endif

