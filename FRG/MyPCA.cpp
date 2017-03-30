#include "MyPCA.h"

#define SHOW_IMAGE 0

MyPCA::MyPCA(vector<string>& _facesPath)
{
	init(_facesPath);
}

void MyPCA::init(vector<string>& _facesPath)
{
    getImgSize(_facesPath);
    imgRows = imread(_facesPath[0],0).rows;
    mergeMatrix(_facesPath);
    getAverageVector();
    subtractMatrix();
    Mat _covarMatrix = (subFacesMatrix.t()) * subFacesMatrix;
    getBestEigenVectors(_covarMatrix);
}

void MyPCA:: getImgSize(vector<string>& _facesPath)
{
    Mat sampleImg = imread(_facesPath[0], 0);
    if (sampleImg.empty()) {
        cout << "Fail to Load Image in PCA" << endl;
    }
    //Dimession of Features
    imgSize = sampleImg.rows * sampleImg.cols;
    //cout << "Per Image Size is: " << size << endl;
}
//put all face images to one matrix, order in column
void MyPCA::mergeMatrix(vector<string>& _facesPath)
{
    int col = int(_facesPath.size());
    allFacesMatrix.create(imgSize, col, CV_32FC1);
    
    for (int i = 0; i < col; i++) {
        Mat tmpMatrix = allFacesMatrix.col(i);
        //Load grayscale image 0
        Mat tmpImg;
        imread(_facesPath[i], 0).convertTo(tmpImg, CV_32FC1);
        //convert to 1D matrix
        tmpImg.reshape(1, imgSize).copyTo(tmpMatrix);
    }
    //cout << "Merged Matix(Width, Height): " << mergedMatrix.size() << endl;
}
//compute average face
void MyPCA::getAverageVector()
{
    //To calculate average face, 1 means that the matrix is reduced to a single column.
    //vector is 1D column vector, face is 2D Mat
    Mat face;
    reduce(allFacesMatrix, avgVector, 1, CV_REDUCE_AVG);
    
    if (SHOW_IMAGE) {
        avgVector.reshape(0, imgRows).copyTo(face);
        //Just for display face
        normalize(face, face, 0, 1, cv::NORM_MINMAX);
        namedWindow("AverageFace", CV_WINDOW_NORMAL);
        imshow("AverageFace", face);
    }
}

void MyPCA::subtractMatrix()
{
    allFacesMatrix.copyTo(subFacesMatrix);
    for (int i = 0; i < subFacesMatrix.cols; i++) {
        subtract(subFacesMatrix.col(i), avgVector, subFacesMatrix.col(i));
    }
}

void MyPCA::getBestEigenVectors(Mat _covarMatrix)
{
    //Get all eigenvalues and eigenvectors from covariance matrix
    Mat allEigenValues, allEigenVectors;
    eigen(_covarMatrix, allEigenValues, allEigenVectors);
    
    eigenVector = allEigenVectors * (subFacesMatrix.t());
    //Normalize eigenvectors
    for(int i = 0; i < eigenVector.rows; i++ )
    {
        Mat tempVec = eigenVector.row(i);
        normalize(tempVec, tempVec);
    }
    
    if (SHOW_IMAGE) {
        //Display eigen face
        Mat eigenFaces, allEigenFaces;
        for (int i = 0; i < eigenVector.rows; i++) {
            eigenVector.row(i).reshape(0, imgRows).copyTo(eigenFaces);
            normalize(eigenFaces, eigenFaces, 0, 1, cv::NORM_MINMAX);
            if(i == 0){
                allEigenFaces = eigenFaces;
            }else{
                hconcat(allEigenFaces, eigenFaces, allEigenFaces);
            }
        }
        
        namedWindow("EigenFaces", CV_WINDOW_NORMAL);
        imshow("EigenFaces", allEigenFaces);
    }
}

Mat MyPCA::getFacesMatrix()
{
    return allFacesMatrix;
}

Mat MyPCA::getAverage()
{
    return avgVector;
}

Mat MyPCA::getEigenvectors()
{
    return eigenVector;
}

MyPCA::~MyPCA() {}
