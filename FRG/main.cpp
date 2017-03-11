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
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace cv;
using namespace std;
using namespace Eigen;


void readFile(string&, vector<string>&, vector<string>&);
int getImgSize(vector<string>&);
Mat mergeMatrix(int, vector<string>&);
Mat subtractMatrix(Mat, Mat);
Mat getAverageVector(Mat, vector<string>&);
Mat getBestEigenVectors(Mat, int);

Mat getBestEigenVectors(Mat allVectors, int largest)
{
    //cout << allVectors.at<float>(0,5059) << endl;
    Mat bestVectors(largest, allVectors.cols, CV_32FC1);
    Mat bestNormalizedVectors(largest, allVectors.cols, CV_32FC1);

    for (int i = 0; i < largest; i++) {
        Mat tmpMatrix = bestVectors.row(i);
        allVectors.row(i).copyTo(tmpMatrix);
        //normalize(allVectors.row(i), allVectors.row(i));
    }
    
    
  /*
    cout << allVectors.at<float>(0,5059) << endl;
    cout << bestVectors.at<float>(0,5059) << endl;
    cout << bestNormalizedVectors.at<float>(0,5059) << endl;
    //cout << "Best Vector" << bestVectors.size() <<endl;
    //cout << "Best Norm Vector" << bestNormalizedVectors.size() <<endl;
 */
    return bestVectors;
}


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
  /*
    Mat covarMatrix, meanMatrix;
    calcCovarMatrix(face, covarMatrix, meanMatrix, CV_COVAR_ROWS, CV_COVAR_SCRAMBLED);

    Mat Values, Vectors;
    eigen(covarMatrix, Values, Vectors);
    cout << "test: " << Vectors.size() << endl;
    for (int i = 0; i < 112; i++) {
        for (int j = 0; j < 112; j++) {
            cout << Vectors.at<float>(i,j) << "  ";
        }
    }
    
 */
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
    string trainListFilePath = "/Users/zichun/Documents/Assignment/FaceRecognition/FRG/train_list.txt";
    vector<string> trainFacesPath;
    vector<int> trainFacesID;
    
    //Load training sets' ID and path to vector
    readFile(trainListFilePath, trainFacesPath, trainFacesID);
    //Get dimession of features for single image
    int imgSize = getImgSize(trainFacesPath);
    //Create a (imgSize X #ofSamples) floating 2D Matrix to store training data
    Mat trainFacesMatrix = mergeMatrix(imgSize, trainFacesPath);
    
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cout << trainFacesMatrix.at<float>(i,j) << " ";
        }
        cout << endl;
    }
    
    //Get average face vector
    Mat trainAvgVector = getAverageVector(trainFacesMatrix, trainFacesPath);
    for (int i = 0; i < 5; i++) {
        cout << trainAvgVector.at<float>(i) << " ||";

    }
    cout << endl;
    //Subtract average face from faces matrix
    Mat subTrainFaceMatrix = subtractMatrix(trainFacesMatrix, trainAvgVector);
    
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cout << subTrainFaceMatrix.at<float>(i,j) << " ";
        }
        cout << endl;
    }
    
    //Get covariance matrix
    Mat covarMatrix(2,2, CV_32FC1), meanMatrix;
    //cout << "ssssss" << endl;
/*    covarMatrix.at<float>(0,0) = 1.2;
    covarMatrix.at<float>(0,1) = 0.8;
    covarMatrix.at<float>(1,0) = 0.8;
    covarMatrix.at<float>(1,1) = 1.2; */
    cout << "ssssss" << endl;
    //calcCovarMatrix(subTrainFaceMatrix, covarMatrix, meanMatrix, CV_COVAR_ROWS, CV_COVAR_SCRAMBLED);
    covarMatrix =  0.2 * (subTrainFaceMatrix * (subTrainFaceMatrix.t()) ) ;
    cout << "Covariance Matrix: "<<covarMatrix.size() << endl;
    //Get all eigenvalues and eigenvectors from covariance matrix
    Mat allEigenVectors;
    Mat allEigenValues;
    eigen(covarMatrix, allEigenValues, allEigenVectors);
/*    cout << "eigenvalues: " << allEigenValues.size() << endl;
    cout << "eigenvalues 0: " << allEigenValues.at<float>(0) << endl;
    cout << "eigenvalues 1: " << allEigenValues.at<float>(1) << endl;
    cout << "eigenvectors: " << allEigenVectors.size() << endl;
    cout << "eigenVECTORS: " << allEigenVectors.at<float>(0,0) << endl;
    cout << "eigenVECTORS: " << allEigenVectors.at<float>(0,1) << endl;
    cout << "eigenVECTORS: " << allEigenVectors.at<float>(1,0) << endl;
    cout << "eigenVECTORS: " << allEigenVectors.at<float>(1,1) << endl;
  */
    /*
    //typedef Matrix<double, Dynamic, Dynamic> covar;
    MatrixXd covar(10304,10304);
    
    for (int i = 0; i < 10304; i++) {
        for (int j = 0; j < 10304; j++) {
            covar(i, j) = covarMatrix.at<float>(i,j);
        }
        //cout << endl;
    }
    cout << "tstdsfgfg: " << covarMatrix.at<float>(0,0) << endl;
    cout << "tstdsfgfg: " << covar(0,0) << endl;
    EigenSolver<MatrixXd> sol_A(covar);
    cout << "fdsfffffff" << endl;
    //sol_A.compute(covar);
    cout << "aaaaaaa" << endl;
    //MatrixXf eigenvals_A =sol_A.eigenvalues().real();
    cout << "bbbbbbbaa" << endl;
    cout << "fsdfdsf " << sol_A.eigenvectors().col(0) << endl;
    */

    
    

    PCA trainPCA;
    trainPCA = PCA(trainFacesMatrix, Mat(), CV_PCA_DATA_AS_COL, 30);
    //cout << "pca: " << trainPCA.eigenvectors << endl;
    
    cout << allEigenValues.at<float>(1) << endl;
    int count = 0;
    printf("%.20f\n", allEigenVectors.at<float>(0,0) );
    for (int i = 0; i < 10304; i++) {
        for (int j = 0; j < 10304; j++) {
            //cout << allEigenVectors.at<float>(i,j) << "  ";
            if (allEigenVectors.at<float>(i,j) > 0.00000000001 || allEigenVectors.at<float>(i,j) < -0.00000000001){
                //cout << i << " .. " << j << endl;
                //cout << allEigenVectors.at<float>(i,j) << "  ";
                count ++;
            }
       
        }
    }
    cout << "xxxxx"<< count << endl;
    //cout << allEigenVectors.at<float>(0,5059) << endl;
    //Keep only k best eigenvectors and normalize
    int largestEigenIndex = 20;
    Mat bestNomEigenVectors = getBestEigenVectors(allEigenVectors, largestEigenIndex);
    cout << bestNomEigenVectors.size() << endl;
    Mat mul;
    //gemm(bestNomEigenVectors, trainFacesMatrix, 1, meanMatrix, 1, mul);
    mul = (subTrainFaceMatrix.t()) * (bestNomEigenVectors.t()) ;
    //multiply(trainFacesMatrix, bestNomEigenVectors, mul);
    cout << mul.size() << endl;
    for (int i = 0; i < mul.rows; i++) {
        for (int j = 0; j < mul.cols; j++) {
           // cout << mul.at<float>(i,j) << " ";
        }
        cout << endl;
    }
    

    
    waitKey();
    return 0;
}
