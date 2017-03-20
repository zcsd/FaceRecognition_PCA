#include <vector>

#include "FaceDetector.h"

FaceDetector::FaceDetector()
{
    String face_cascadePath = "/Users/zichun/Documents/Assignment/FaceRecognition/FRG/haarcascade/haarcascade_frontalface_alt.xml";
    String eyes_cascadePath = "/Users/zichun/Documents/Assignment/FaceRecognition/FRG/haarcascade/haarcascade_eye.xml";
    //haarcascade_eye_tree_eyeglasses
    
    if ( !face_cascade.load(face_cascadePath) )
        cout << "ERROR:***Can not load face cascade***" << endl;
    
    if ( !eye_cascade.load(eyes_cascadePath) )
        cout << "ERROR:***Can not load eye cascade***" << endl;
    
}

void FaceDetector::findFacesInImage(Mat &frameRGB) {
    Mat frameGray;

    //convert the image to grayscale and normalize histogram:
    resize(frameRGB, frameRGB, Size(320, 240));
    cvtColor(frameRGB, frameGray, CV_BGR2GRAY);
    equalizeHist(frameGray, frameGray);
    
    vector<Rect> facesRec;
    
    //detect faces:
    face_cascade.detectMultiScale( frameGray, facesRec, 1.1, 5, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    cout << "faces: " << facesRec.size() << endl;
    
    if (facesRec.size() >= 1){
        rectangle(frameRGB, facesRec[0], Scalar( 255, 0, 255 ), 4);
        
        Mat faceROI = frameGray(facesRec[0]);
        cout << "ROI SIZE " << faceROI.size() << endl;
        vector<Rect> eyes;
        
        //detect eyes
        eye_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
        cout << "eyes: " << eyes.size() << endl;
        
        for ( size_t j = 0; j < eyes.size(); j++ )
        {
            Point eye_center( facesRec[0].x + eyes[j].x + eyes[j].width/2, facesRec[0].y + eyes[j].y + eyes[j].height/2 );
            circle(frameRGB, eye_center, 2, Scalar( 255, 0, 0 ), 4, 8, 0);
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
