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
    
    Mat avgVec, eigenVec, facesInEigen;
    facesInEigen =  readFaces(int(trainFacesID.size()));
    avgVec = readMean();
    eigenVec = readEigen(int(trainFacesID.size()));
    
    Mat frame, processed,testImg;
    namedWindow("Face Recognisation", CV_WINDOW_NORMAL);
    
    cout << "++++++Welcome to Face Recognisation System++++++" << endl;
    cout << "Prepare Faces(0) or Training(1) or Recognise(2), Input your number:  ";
    int choice;
    scanf("%d", &choice);
    
    if (choice == 0) {
        cout << "Prepare Face Start......" << endl;
        //Initialize capture
        GetFrame getFrame(1);
        getFrame.getNextFrame(frame);
        
        //TO DO FACE DETECTION
        FaceDetector faceDetector;
        faceDetector.findFacesInImage(frame, processed);
        
        if (faceDetector.goodFace()) {
            testImg = faceDetector.getFaceToTest();
            imwrite("/Users/zichun/Documents/Assignment/FaceRecognition/FRG/s1.bmp", testImg);
        }
        cout << "Prepare Face Finished." << endl;
        imshow("Face Recognisation", processed);
        waitKey();
        //Prepare finish.
    }else if (choice == 1){
        cout << "Traning Start......" << endl;
        //do PCA analysis for training faces
        MyPCA myPCA = MyPCA(trainFacesPath);
        //Write trainning data to file
        WriteTrainData wtd = WriteTrainData(myPCA, trainFacesID);
        //training finsih.
        cout << "Training finsih." << endl;
    }else if (choice == 2){
        cout << "Recognise Start......" << endl;
        //Initialize capture
        GetFrame getFrame(1);
        getFrame.getNextFrame(frame);
        
        //TO DO FACE DETECTION
        FaceDetector faceDetector;
        faceDetector.findFacesInImage(frame, processed);
        
        if (faceDetector.goodFace()) {
            testImg = faceDetector.getFaceToTest();
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
            
            imshow("Face Recognisation", processed);
            waitKey();
        }else{
            cout << "Face detection not good" << endl;
        }
        
    }else{
        cout << "Input wrong choice......" << endl;
    }
   
    return 0;
}
