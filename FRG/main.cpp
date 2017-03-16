#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "GetFrame.h"
#include "MyPCA.h"
#include "FaceRecognizer.h"

using namespace cv;
using namespace std;

void readFile(string&, vector<string>&, vector<string>&);
Mat getProjectedFaces(Mat, Mat, Mat);
int recogniseFace(Mat, vector<Mat>&, vector<int>&);

int recogniseFace(Mat testFace, vector<Mat>& trainFaces, vector<int>& trainFacesID)
{
    double closetFaceDist = 4000;
    int closetFaceID = -1;
    
    for (int i =0; i < trainFaces.size(); i++) {
        Mat src1 = trainFaces[i];
        Mat src2 = testFace;
        
        double dist = norm(src1, src2, NORM_L2);
        //cout << dist << endl;
        
        if (dist < closetFaceDist) {
            closetFaceDist = dist;
            closetFaceID = trainFacesID[i];
        }
    }
    cout << "Closet Distance: " <<closetFaceDist << endl;
    
    return closetFaceID;
}

Mat getProjectedFaces(Mat inputFaceVec, Mat meanFaceVec, Mat eigenVecs){
    Mat prjFace, tmpData;
    
    if (inputFaceVec.cols != 1 || meanFaceVec.cols != 1 || inputFaceVec.rows != meanFaceVec.rows) {
        cout << "Wrong Input in getProjectedFaces!" << endl;
        exit(1);
    }
    subtract(inputFaceVec, meanFaceVec, tmpData);
    prjFace = eigenVecs * tmpData;
    //cout << "Projected Face(W, H):" <<prjFace.size() << endl;
    //cout << prjFace.at<float>(0) << endl;
    
    return prjFace;
}

void readFile(string& listFilePath, vector<string>& facesPath, vector<int>& facesID)
{
    ifstream file(listFilePath.c_str(), ifstream::in);
    
    if (!file) {
        cout << "Fail to open file: " << listFilePath << endl;
        exit(0);
    }
    
    string line, path, id;
    while (getline(file, line)) {
        stringstream lines(line);
        getline(lines, id, ';');
        getline(lines, path);
        
        path.erase(remove(path.begin(), path.end(), '\r'), path.end());
        path.erase(remove(path.begin(), path.end(), '\n'), path.end());
        path.erase(remove(path.begin(), path.end(), ' '), path.end());
        
        facesPath.push_back(path);
        facesID.push_back(atoi(id.c_str()));
    }
    
    for(int i = 0; i < facesPath.size(); i++) {
        cout << facesID[i] << " : " << facesPath[i] << endl;
    }
}

int main(int argc, char** argv)
{
    Mat frame;
    namedWindow("Face Recognisation", CV_WINDOW_NORMAL);
    //Initialize capture
    GetFrame getFrame(1);
    getFrame.getNextFrame(frame);
    imshow("Face Recognisation", frame);

    string trainListFilePath = "/Users/zichun/Documents/Assignment/FaceRecognition/FRG/train_list.txt";
    vector<string> trainFacesPath;
    vector<int> trainFacesID;
    
    //Load testing image
    Mat testImg = imread("/Users/zichun/Documents/Assignment/FaceRecognition/DerivedData/FaceRecognition/Build/Products/Debug/att_faces/s2/8.pgm",0);

    //Load training sets' ID and path to vector
    readFile(trainListFilePath, trainFacesPath, trainFacesID);
    
    MyPCA myPCA = MyPCA(trainFacesPath);
    //Project training face to eigen space
    vector<Mat> baseFaces;
    for (int i = 0; i < trainFacesPath.size(); i++){
        baseFaces.push_back(getProjectedFaces(myPCA.getFacesMatrix().col(i), myPCA.getAverage(), myPCA.getEigenvectors() ) );
    }
    
    //reshape input test image
    Mat testVec;
    testImg.convertTo(testImg, CV_32FC1);
    testImg.reshape(0, 10304).copyTo(testVec);
    //Project test face to eigen space
    Mat testFace = getProjectedFaces(testVec, myPCA.getAverage(), myPCA.getEigenvectors());
    //Face Recognisation and get result
    int testResultFaceID = recogniseFace(testFace, baseFaces, trainFacesID);
    if (testResultFaceID != -1) {
        cout << "Face ID: " << testResultFaceID << endl;
    }else{
        cout << "Unknown Face." << endl;
    }
    
    waitKey();
    return 0;
}
