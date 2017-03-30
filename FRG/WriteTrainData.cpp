#include "WriteTrainData.h"

WriteTrainData::WriteTrainData(MyPCA _trainPCA, vector<string>& _trainFacesID)
{
    numberOfFaces = _trainPCA.getFacesMatrix().cols;
    trainFacesInEigen.create(numberOfFaces, numberOfFaces, CV_32FC1);
    project(_trainPCA);
    writeTrainFacesData(_trainFacesID);
    writeMean(_trainPCA.getAverage());
    writeEigen(_trainPCA.getEigenvectors());
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
    //cout << trainFacesInEigen.col(0).size() <<endl;
}

void WriteTrainData::writeTrainFacesData(vector<string>& _trainFacesID)
{
    string facesDataPath = "/Users/zichun/Documents/Assignment/FaceRecognition/FRG/data/facesdata.txt";
    ofstream writeFaceFile(facesDataPath.c_str(), ofstream::out | ofstream::trunc);
    if (!writeFaceFile) {
        cout << "Fail to open file: " << facesDataPath << endl;
    }
    
    for (int i = 0; i < _trainFacesID.size(); i++) {
        //writeFaceFile << i + 1 << "#";
        writeFaceFile << _trainFacesID[i] << ":";
        for (int j = 0; j < trainFacesInEigen.rows; j++) {
            writeFaceFile << trainFacesInEigen.col(i).at<float>(j);
            writeFaceFile << " ";
        }
        writeFaceFile << "\n";
    }
    
    writeFaceFile.close();
}

void WriteTrainData::writeMean(Mat avg)
{
    string meanPath = "/Users/zichun/Documents/Assignment/FaceRecognition/FRG/data/mean.txt";
    ofstream writeMeanFile(meanPath.c_str(), ofstream::out | ofstream::trunc);
    if (!writeMeanFile) {
        cout << "Fail to open file: " << meanPath << endl;
    }
    
    for (int i = 0; i < avg.rows; i++) {
        writeMeanFile << avg.at<float>(i);
        writeMeanFile << " ";
    }
    
    writeMeanFile.close();
}

void WriteTrainData::writeEigen(Mat eigen)
{
    string eigenPath = "/Users/zichun/Documents/Assignment/FaceRecognition/FRG/data/eigen.txt";
    ofstream writeEigenFile(eigenPath.c_str(), ofstream::out | ofstream::trunc);
    if (!writeEigenFile) {
        cout << "Fail to open file: " << eigenPath << endl;
    }
    
    for (int i = 0; i < eigen.rows; i++) {
        for (int j = 0; j < eigen.cols; j++) {
            writeEigenFile << eigen.row(i).at<float>(j);
            writeEigenFile << " ";
        }
        writeEigenFile << "\n";
    }
    
    writeEigenFile.close();
}

Mat WriteTrainData::getFacesInEigen()
{
    return trainFacesInEigen;
}

WriteTrainData::~WriteTrainData() {}
