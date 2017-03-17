#include "WriteTrainData.h"

WriteTrainData::WriteTrainData(MyPCA _trainPCA)
{
    numberOfFaces = _trainPCA.getFacesMatrix().cols;
    trainFacesInEigen.create(numberOfFaces, numberOfFaces, CV_32FC1);
    project(_trainPCA);
}

void WriteTrainData::project(MyPCA _trainPCA)
{
    //cout << "Write Class"<<_trainPCA.getFacesMatrix().size() << endl;
    Mat facesMatrix = _trainPCA.getFacesMatrix();
    Mat avg = _trainPCA.getAverage();
    Mat eigenVec = _trainPCA.getEigenvectors();
    
    for (int i = 0; i < numberOfFaces; i++) {
        Mat temp;
        Mat projectFace = trainFacesInEigen.col(i);
        subtract(facesMatrix.col(i), avg, temp);
        projectFace = eigenVec * temp;
    }
}

void WriteTrainData::writeTrainFacesData()
{
    
}

Mat WriteTrainData::getFacesInEigen()
{
    return trainFacesInEigen;
}

WriteTrainData::~WriteTrainData() {}
