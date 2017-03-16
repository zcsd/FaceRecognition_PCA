#include "GetFrame.h"

GetFrame::GetFrame(bool startFrame):_startFrame(startFrame)
{
    //capture from default webcam
    _vid.open(0);
    
    if ( _vid.isOpened() ) {
        _vid.set(CV_CAP_PROP_FRAME_WIDTH, 640);
        _vid.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
        cout << "Initialize Capture(640x480) Successfully. " << endl;
    }else{
        cout << "ERROR: ***Could not initialize capturing***" << endl;
    }
}

bool GetFrame::getNextFrame(Mat &img) {
    if ( !_vid.read(img) ) {
        cout << "ERROR: ***Could not read frame***" << endl;
        return 0;
    }else{
        return 1;
    }
}

GetFrame::~GetFrame() {
    _vid.release();
}
