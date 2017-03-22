#include <stdio.h>
#include <vector>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "ReadFile.h"
#include "GetFrame.h"
#include "MyPCA.h"
#include "FaceRecognizer.h"
#include "WriteTrainData.h"
#include "FaceDetector.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    string trainListFilePath = "/Users/zichun/Documents/Assignment/FaceRecognition/FRG/list/train_list.txt";
    vector<string> trainFacesPath;
    vector<string> trainFacesID;
    readList(trainListFilePath, trainFacesPath, trainFacesID);
    /*
    cout << "++++++Welcome to Face Recognisation System++++++" << endl;
    cout << "Do you want to do training?(Y/N): ";
    char choice;
    scanf("%s", &choice);
    
    if (choice == 'Y' || choice == 'y') {
        cout << "Training Start......" << endl;
        
    }else{
        cout << "Recognise Start......" << endl;
        
    }
    */
    Mat frame;
    namedWindow("Face Recognisation", CV_WINDOW_NORMAL);
    //Initialize capture
    GetFrame getFrame(1);
    getFrame.getNextFrame(frame);
    
    //TO DO FACE DETECTION
    FaceDetector faceDetector;
    faceDetector.findFacesInImage(frame);
    
    Mat testImg;
    if (faceDetector.goodFace()) {
        //testImg = imread("/Users/zichun/Documents/Assignment/FaceRecognition/FRG/faces/02/s6.bmp",0);
        testImg = faceDetector.getFaceToTest();
        imwrite("/Users/zichun/Documents/Assignment/FaceRecognition/FRG/s1.bmp", testImg);
    }else{
        testImg = imread("/Users/zichun/Documents/Assignment/FaceRecognition/FRG/faces/02/s8.bmp",0);
    }
    
    //do PCA analysis for training faces
    MyPCA myPCA = MyPCA(trainFacesPath);
    //Write trainning data to file
    WriteTrainData wtd = WriteTrainData(myPCA, trainFacesID);
    /////////////////////////Load data
    bool flag = 1;
    Mat avgVec, eigenVec, facesInEigen;
    if ( flag ) {
        facesInEigen =  readFaces(int(trainFacesID.size()));
        avgVec = readMean();
        eigenVec = readEigen(int(trainFacesID.size()));
    }else{
        avgVec = myPCA.getAverage();
        eigenVec = myPCA.getEigenvectors();
        facesInEigen = wtd.getFacesInEigen();
    }
    
    //final step: recognize new face from training faces
    FaceRecognizer faceRecognizer = FaceRecognizer(testImg, avgVec, eigenVec, facesInEigen, trainFacesID);
    // Show Result
    string faceID = faceRecognizer.getClosetFaceID();
    if (faceID != "None") {
        cout << "Face ID: " << faceID;
        cout << "   Distance: " << faceRecognizer.getClosetDist() << endl;
    }else{
        cout << "Unkown Face" << endl;
    }
    
    imshow("Face Recognisation", frame);
    
    waitKey();
    return 0;
}
