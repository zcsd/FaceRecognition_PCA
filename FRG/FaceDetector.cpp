#include <vector>

#include "FaceDetector.h"

FaceDetector::FaceDetector()
{
    string cascadePath = "/Users/zichun/Documents/Assignment/FaceRecognition/FRG/haarcascade/haarcascade_frontalface_alt.xml";
    _scaleFactor = 1.01;
    _minNeighbors = 40;
    _minSizeRatio = 0.06;
    _maxSizeRatio = 0.18;
    
    if ( !_cascade.load(cascadePath) )
        cout << "ERROR:***Can not load cascade***" << endl;
    
}

void FaceDetector::findFacesInImage(Mat img, vector<Rect> &res) {
    Mat tmp;
    int width  = img.size().width,
        height = img.size().height;
    Size minScaleSize = Size(_minSizeRatio  * width, _minSizeRatio  * height),
         maxScaleSize = Size(_maxSizeRatio  * width, _maxSizeRatio  * height);
    
    //convert the image to grayscale and normalize histogram:
    cvtColor(img, tmp, CV_BGR2GRAY);
    equalizeHist(tmp, tmp);
    
    //clear the vector:
    res.clear();
    
    //detect faces:
    _cascade.detectMultiScale(tmp, res, _scaleFactor, _minNeighbors, 0, minScaleSize, maxScaleSize);
}

FaceDetector::~FaceDetector() {}
