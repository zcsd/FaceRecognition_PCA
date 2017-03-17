#ifndef READ_LIST_H
#define READ_LIST_H

#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

void readList(string& listFilePath, vector<string>& facesPath, vector<int>& facesID)
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
    /*
    for(int i = 0; i < facesPath.size(); i++)
        cout << facesID[i] << " : " << facesPath[i] << endl;
     */
    
}

#endif
