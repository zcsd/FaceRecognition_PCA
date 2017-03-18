#ifndef FACE_DETECTOR_H
#define	FACE_DETECTOR_H

#include <string>
#include <iostream>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core.hpp"

using namespace std;
using namespace cv;

class FaceDetector {
public:
    FaceDetector();
    ~FaceDetector();
    void findFacesInImage(Mat img, vector<Rect> &res);
private:
    CascadeClassifier _cascade;
    double _scaleFactor;
    int    _minNeighbors;
    double _minSizeRatio;
    double _maxSizeRatio;
};

#endif

