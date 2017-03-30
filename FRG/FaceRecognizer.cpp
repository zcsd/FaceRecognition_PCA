#include "FaceRecognizer.h"

FaceRecognizer::FaceRecognizer(Mat _testImg, Mat _avgVec, Mat _eigenVec, Mat _facesInEigen, vector<string>& _loadedFacesID) {
    prepareFace(_testImg);
    projectFace(testVec, _avgVec, _eigenVec);
    recognize(testPrjFace, _facesInEigen, _loadedFacesID);
    
}

void FaceRecognizer::prepareFace(Mat _testImg)
{
    _testImg.convertTo(_testImg, CV_32FC1);
    _testImg.reshape(0, _testImg.rows*_testImg.cols).copyTo(testVec);
}

void FaceRecognizer::projectFace(Mat testVec, Mat _avgVec, Mat _eigenVec){
    Mat tmpData;
    
    subtract(testVec, _avgVec, tmpData);
    testPrjFace = _eigenVec * tmpData;
}
//Find the closet Euclidean Distance between input and database
void FaceRecognizer::recognize(Mat testPrjFace, Mat _facesInEigen, vector<string>& _loadedFacesID)
{
    for (int i =0; i < _loadedFacesID.size(); i++) {
        Mat src1 = _facesInEigen.col(i);
        Mat src2 = testPrjFace;
        
        double dist = norm(src1, src2, NORM_L2);
        //cout << "Dist " <<dist << endl;
        if (dist < closetFaceDist) {
            closetFaceDist = dist;
            closetFaceID = _loadedFacesID[i];
        }
    }
    //cout  << "id " << closetFaceID << endl;
    //cout << "Closet Distance: " << closetFaceDist << endl;
}

string FaceRecognizer::getClosetFaceID()
{
    return closetFaceID;
}

double FaceRecognizer::getClosetDist()
{
    return closetFaceDist;
}

FaceRecognizer::~FaceRecognizer() {}
