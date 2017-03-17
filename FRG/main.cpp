#include <stdio.h>
#include <vector>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "ReadList.h"
#include "GetFrame.h"
#include "MyPCA.h"
#include "FaceRecognizer.h"
#include "WriteTrainData.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat frame;
    namedWindow("Face Recognisation", CV_WINDOW_NORMAL);
    //Initialize capture
    GetFrame getFrame(1);
    getFrame.getNextFrame(frame);
    imshow("Face Recognisation", frame);
    
    //TO Load Training Data
    
    //TO DO FACE DETECTION
    
    //TO prepare test face

    string trainListFilePath = "/Users/zichun/Documents/Assignment/FaceRecognition/FRG/train_list.txt";
    vector<string> trainFacesPath;
    vector<int> trainFacesID;
    
    //Tempory using....Load testing image
    Mat testImg = imread("/Users/zichun/Documents/Assignment/FaceRecognition/FRG/att_faces/s2/9.pgm",0);

    //Load training sets' ID and path to vector
    readList(trainListFilePath, trainFacesPath, trainFacesID);
    //do PCA analysis for training faces
    MyPCA myPCA = MyPCA(trainFacesPath);
    //Write trainning data to file
    WriteTrainData wtd = WriteTrainData(myPCA);
    //final step: recognize new face from training faces
    Mat avgVec = myPCA.getAverage();
    Mat eigenVec = myPCA.getEigenvectors();
    Mat facesInEigen = wtd.getFacesInEigen();
    FaceRecognizer faceRecognizer = FaceRecognizer(testImg, avgVec, eigenVec, facesInEigen, trainFacesID);
    // Show Result
    int faceID = faceRecognizer.getClosetFaceID();
    if (faceID != -1) {
        cout << "Face ID: " << faceID;
        cout << "   Distance: " << faceRecognizer.getClosetDist() << endl;
    }else{
        cout << "Unkown Face" << endl;
    }

    waitKey();
    return 0;
}
