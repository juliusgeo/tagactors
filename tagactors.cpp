#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
//#include <iostream>
 using namespace std;
 using namespace cv;
int main(int argc, char** argv)
{
    String filename = "yourfile.avi";
    VideoCapture capture(filename);
    Mat frame;

    if( !capture.isOpened() )
        throw "Error when reading steam_avi";

    void detect_faces (IplImage*, CvHaarClassifierCascade*, CvMemStorage*);
    void cleanup (char*, CvHaarClassifierCascade*, CvMemStorage*);



    char *classifer = "/haarcascade_frontalface_default.xml";
    namedWindow( "w", 1);
    for( ; ; )
    {
        capture >> frame;
        if(frame.empty())
            break;
        CvHaarClassifierCascade* cascade = 0;
        CvMemStorage* storage = 0;
        char* window_name = "haar window";
        //initialize
        cv::Mat image1;
        image1=frame;
        IplImage* image2;
        image2 = cvCreateImage(cvSize(image1.cols,image1.rows),8,3);
        IplImage ipltemp=image1;
        cvCopy(&ipltemp,image2);

        cascade = (CvHaarClassifierCascade*) cvLoad(classifer, 0, 0, 0 );
        storage = cvCreateMemStorage(0);
        assert(cascade && storage);
        detect_faces(image2, cascade, storage); //detect and draw
        cv::Mat m = cv::cvarrToMat(image2, true);
        cv::imshow(window_name, m);
        cv::waitKey(10); // waits to display frame
    }
    waitKey(0); // key press to close window
    // releases and window destroy are automatic in C++ interface
}

void detect_faces (IplImage* image,
                   CvHaarClassifierCascade* cascade,
                   CvMemStorage* storage) {

  //get a sequence of faces in image
  CvSeq *faces = cvHaarDetectObjects(image, cascade, storage,
     1.1,                       //increase search scale by 10% each pass
     3,                         //drop groups of fewer than n detections
     CV_HAAR_DO_CANNY_PRUNING,  //skip regions unlikely to contain a face
     cvSize(0,0));              //smallest face to search for, use XML default

  //draw rectangle outline around each detection
  int i;
  for(i = 0; i < (faces ? faces->total : 0); i++ ) {
    CvRect *r = (CvRect*) cvGetSeqElem(faces, i);
    CvPoint top_left = { r->x, r->y };
    CvPoint bot_right = { r->x + r->width, r->y + r->height };
    cvRectangle(image, top_left, bot_right, CV_RGB(0,255,0), 3, 4, 0);
  }
}


void cleanup (char* name,
              CvHaarClassifierCascade* cascade,
              CvMemStorage* storage) {
  //cleanup and release resources
  cvDestroyWindow(name);
  if(cascade) cvReleaseHaarClassifierCascade(&cascade);
  if(storage) cvReleaseMemStorage(&storage);
}