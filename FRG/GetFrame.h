/*Capture frame from webcam*/
#ifndef GET_FRAME_H
#define GET_FRAME_H

#include <iostream>
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

class GetFrame
{
public:
    GetFrame(bool startFrame = 1);
    ~GetFrame();
    bool getNextFrame(Mat &frame);
private:
    VideoCapture _vid;
    bool _startFrame;
};

#endif

