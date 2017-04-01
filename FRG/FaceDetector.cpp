#include <vector>

#include "FaceDetector.h"

FaceDetector::FaceDetector()
{
    String face_cascadePath = "/Users/zichun/Documents/Assignment/FaceRecognition/FRG/haarcascade/haarcascade_frontalface_default.xml";
    String eyes_cascadePath = "/Users/zichun/Documents/Assignment/FaceRecognition/FRG/haarcascade/haarcascade_eye.xml";
    //haarcascade_eye_tree_eyeglasses
    
    if ( !face_cascade.load(face_cascadePath) )
        cout << "ERROR:***Can not load face cascade***" << endl;
    
    if ( !eye_cascade.load(eyes_cascadePath) )
        cout << "ERROR:***Can not load eye cascade***" << endl;
    
}
//Eye detection accuracy is very bad, so don't check the rotation angle
void FaceDetector::findFacesInImage(Mat &frameRGB, Mat &toTest) {
    Mat frameGray;
    
    toTest = Mat::zeros(480, 480, frameRGB.type());
    for (int i = 0; i < toTest.cols; i++) {
        frameRGB.col(80 + i).copyTo(toTest.col(i));
    }
    
    //convert the image to grayscale and normalize histogram:
    resize(toTest, toTest, Size(240, 240));
    cvtColor(toTest, frameGray, CV_BGR2GRAY);
    Mat toReturn;
    frameGray.copyTo(toReturn);
    //cout << toTest.size() << endl;
    equalizeHist(frameGray, frameGray);
    
    vector<Rect> facesRec;
    
    //detect faces:
    face_cascade.detectMultiScale( frameGray, facesRec, 1.1, 5, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    //cout << "faces: " << facesRec.size() << endl;
    
    if (facesRec.size() >= 1){
        rectangle(toTest, facesRec[0], Scalar( 255, 0, 255 ), 4);
        
        Mat faceROI = toReturn(facesRec[0]);
        //cout << "ROI SIZE " << faceROI.size() << endl;
        faceROI.copyTo(faceToTest);
        resize(faceToTest, faceToTest, Size(100,100));
        faceFlag = 1;
        
        vector<Rect> eyes;
        //detect eyes
        eye_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
        //cout << "eyes: " << eyes.size() << endl;
        
        for ( size_t j = 0; j < eyes.size(); j++ )
        {
            Point eye_center( facesRec[0].x + eyes[j].x + eyes[j].width/2, facesRec[0].y + eyes[j].y + eyes[j].height/2 );
            circle(toTest, eye_center, 2, Scalar( 255, 0, 0 ), 4, 8, 0);
        }
        
        eyes.clear();
    }
    
    facesRec.clear();
}

bool FaceDetector::goodFace()
{
    return faceFlag;
}

Mat FaceDetector::getFaceToTest()
{
    return faceToTest;
}

FaceDetector::~FaceDetector() {}
