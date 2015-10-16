#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include <stdio.h>
#include <iostream>
//#include <stdio.h>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    int numBoards = atoi(argv[1]);
    int board_w = atoi(argv[2]);
    int board_h = atoi(argv[3]);

    Size board_sz = Size(board_w, board_h);
    int board_n = board_w*board_h;

    vector<vector<Point3f> > object_points;
    vector<vector<Point2f> > image_points;
    vector<Point2f> corners;

    vector<Point3f> obj;
    for (int j=0; j<board_n; j++)
    {
        obj.push_back(Point3f(j/board_h, j%board_h, 0.0f));
    }

    Mat img, gray;
     cout << "Device opening ..." << endl;
    VideoCapture capture;

    capture.open( CV_CAP_OPENNI );
        
    //registration
    if(capture.get( CV_CAP_PROP_OPENNI_REGISTRATION ) == 0) 
    {
        capture.set(CV_CAP_PROP_OPENNI_REGISTRATION, 1);
   
        cout << "\nImages have been registered ..." << endl;
    }
    //cout << cv::getBuildInformation() << endl;

    if( !capture.isOpened() )
    {
        cout << "Can not open a capture object." << endl;
        return -1;
    }

    if( capture.get( CV_CAP_OPENNI_IMAGE_GENERATOR_PRESENT ) )
    {
        cout <<
            "\nImage generator output mode:" << endl <<
            "FRAME_WIDTH   " << capture.get( CV_CAP_OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_FRAME_WIDTH ) << 
            "   | FRAME_HEIGHT  " << capture.get( CV_CAP_OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_FRAME_HEIGHT ) << 
            "   | FPS           " << capture.get( CV_CAP_OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_FPS ) << endl;
    }
    else
    {
        cout << "\nDevice doesn't contain image generator." << endl;
    }  

    int success = 0;
    int k = 0;
    bool found = false;
    while (success < numBoards)
    {
         if( !capture.grab() )
        {
            cout << "Can not grab images." << endl;
            return -1;
        }
            //capture.retrieve( depth, CV_CAP_OPENNI_DEPTH_MAP );
            
            capture.retrieve( img, CV_CAP_OPENNI_BGR_IMAGE );
        cvtColor(img, gray, CV_BGR2GRAY);
        found = findChessboardCorners(gray, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);

        if (found)
        {
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(gray, board_sz, corners, found);
        }

       // imshow("image", img);
        imshow("corners", gray);

        k = waitKey(1);
        if (found)
        {
            k = waitKey(0);
        }
        if (k == 27)
        {
            break;
        }
        if (k == ' ' && found !=0)
        {
            image_points.push_back(corners);
            object_points.push_back(obj);
            printf ("Corners stored\n");
            success++;

            if (success >= numBoards)
            {
                break;
            }
        }

    }
    destroyAllWindows();
    printf("Starting calibration\n");
    Mat intrinsic = Mat(3, 3, CV_32FC1);
    Mat distcoeffs;
    vector<Mat> rvecs, tvecs;

    intrinsic.at<float>(0, 0) = 1;
    intrinsic.at<float>(1, 1) = 1;
    
    calibrateCamera(object_points, image_points, img.size(), intrinsic, distcoeffs, rvecs, tvecs);

    FileStorage fs1("mycalib.yml", FileStorage::WRITE);
    fs1 << "CM1" << intrinsic;
    fs1 << "D1" << distcoeffs;

    printf("calibration done\n");

//    Mat imgU;
//    while(1)
//    {
//        cap >> img;
//        undistort(img, imgU, intrinsic, distcoeffs);
//
//        imshow("image", img);
//        imshow("undistort", imgU);
//
//        k = waitKey(5);
//        if (k == 27)
//        {
//            break;
//        }
//    }
//    cap.release();
    return(0);
}