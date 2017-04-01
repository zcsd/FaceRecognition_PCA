/*Recognise new input face from database*/
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
    FaceRecognizer(Mat _testImg, Mat _avgVec, Mat _eigenVec, Mat _facesInEigen, vector<string>& _loadedFacesID);
    void prepareFace(Mat _testImg);
    void projectFace(Mat testVec, Mat _avgVec, Mat _eigenVec);
    void recognize(Mat testPrjFace, Mat _facesInEigen, vector<string>& _loadedFacesID);
    string getClosetFaceID();
    double getClosetDist();
    ~FaceRecognizer();

private:
    Mat testVec;
    Mat testPrjFace;
    string closetFaceID = "None";
    double closetFaceDist = 3500;
};

#endif

