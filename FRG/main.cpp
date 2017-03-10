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

using namespace cv;
using namespace std;

void readFile(string&, vector<string>&, vector<string>&);
int getImgSize(vector<string>&);
Mat mergeMatrix(int, vector<string>&);
Mat subtractMatrix(Mat, Mat);
Mat getAverageVector(Mat, vector<string>&);

Mat getAverageVector(Mat facesMatrix, vector<string>& facesPath)
{
    //To calculate average face, 1 means that the matrix is reduced to a single column.
    //vector is 1D column vector, face is 2D Mat
    Mat vector, face;
    reduce(facesMatrix, vector, 1, CV_REDUCE_AVG);
    vector.reshape(0, imread(facesPath[0],0).rows).copyTo(face);
    //Just for display face
    normalize(face, face, 0, 1, cv::NORM_MINMAX);
    namedWindow("Average Face", CV_WINDOW_NORMAL);
    imshow("Average Face", face);
    
    return vector;
}

Mat subtractMatrix(Mat facesMatrix, Mat avgVector)
{
    Mat resultMatrix;
    facesMatrix.copyTo(resultMatrix);
    for (int i = 0; i < resultMatrix.cols; i++) {
        subtract(resultMatrix.col(i), avgVector, resultMatrix.col(i));
    }
    return resultMatrix;
}

Mat mergeMatrix(int row, vector<string>& facesPath)
{
    int col = int(facesPath.size());
    Mat mergedMatrix(row, col, CV_32FC1);
    
    for (int i = 0; i < col; i++) {
        Mat tmpMatrix = mergedMatrix.col(i);
        //Load grayscale image 0
        Mat tmpImg;
        imread(facesPath[i], 0).convertTo(tmpImg, CV_32FC1);
        //convert to 1D matrix
        tmpImg.reshape(1, row).copyTo(tmpMatrix);
    }
    cout << "Merged Matix(Width, Height): " << mergedMatrix.size() << endl;

    return mergedMatrix;
}

int getImgSize(vector<string>& facesPath)
{
    Mat sampleImg = imread(facesPath[0], 0);
    if (sampleImg.empty()) {
        cout << "Fail to Load Image" << endl;
        exit(1);
    }
    //Dimession of Features
    int size = sampleImg.rows * sampleImg.cols;
    //cout << "Per Image Size is: " << size << endl;
    return size;
}

void readFile(string& listFilePath, vector<string>& facesPath, vector<int>& facesID)
{
    std::ifstream file(listFilePath.c_str(), ifstream::in);
    
    if (!file) {
        cout << "Fail to open file: " << listFilePath << endl;
        exit(0);
    }
    
    std::string line, path, id;
    while (std::getline(file, line)) {
        std::stringstream lines(line);
        std::getline(lines, id, ';');
        std::getline(lines, path);
        
        path.erase(std::remove(path.begin(), path.end(), '\r'), path.end());
        path.erase(std::remove(path.begin(), path.end(), '\n'), path.end());
        path.erase(std::remove(path.begin(), path.end(), ' '), path.end());
        
        facesPath.push_back(path);
        facesID.push_back(atoi(id.c_str()));
    }
    
    for(int i = 0; i < facesPath.size(); i++) {
        cout << facesID[i] << " : " << facesPath[i] << endl;
    }
}

int main(int argc, char** argv)
{
    string trainListFilePath = "/Users/zichun/Documents/Assignment/FaceRecognition/FRG/train_list.txt";
    vector<string> trainFacesPath;
    vector<int> trainFacesID;
    
    //Load training sets' ID and path to vector
    readFile(trainListFilePath, trainFacesPath, trainFacesID);
    //Get dimession of features for single image
    int imgSize = getImgSize(trainFacesPath);
    //Create a (imgSize X #ofSamples) floating 2D Matrix to store training data
    Mat trainFacesMatrix = mergeMatrix(imgSize, trainFacesPath);
    //Get average face vector
    Mat trainAvgVector = getAverageVector(trainFacesMatrix, trainFacesPath);
    //Subtract average face from faces matrix
    Mat subTrainFaceMatrix = subtractMatrix(trainFacesMatrix, trainAvgVector);

    /*
    PCA trainPCA;
    trainPCA = PCA(trainFacesMatrix, Mat(), CV_PCA_DATA_AS_COL, 30);
    cout << "mean: " << trainPCA.eigenvalues << endl;
     */
    
    waitKey();
    return 0;
}
