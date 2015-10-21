#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    //int numBoards = atoi(argv[1]);
    int board_w = atoi(argv[1]);
    int board_h = atoi(argv[2]);
    const char* rgb_intrinsics = argv[3];
    const char* depth_intrinsics = argv[4];
    
    Size board_sz = Size(board_w, board_h);
    int board_n = board_w*board_h;

    vector<vector<Point3f> > object_points;
    vector<vector<Point2f> > imagePoints1, imagePoints2;
    vector<Point2f> corners1, corners2;

    vector<Point3f> obj;
    for (int j=0; j<board_n; j++)
    {
        obj.push_back(Point3f(j/board_w, j%board_w, 0.0f));
    }

    Mat img1, img2, gray1, gray2;
    
    VideoCapture capture1;

    capture1.open( CV_CAP_OPENNI );
        
    //registration
    if(capture1.get( CV_CAP_PROP_OPENNI_REGISTRATION ) == 0) 
    {
        capture1.set(CV_CAP_PROP_OPENNI_REGISTRATION, 1);
   
        cout << "\nImages have been registered ..." << endl;
    }
    //cout << cv::getBuildInformation() << endl;

    if( !capture1.isOpened() )
    {
        cout << "Can not open a capture object." << endl;
        return -1;
    }

    if( capture1.get( CV_CAP_OPENNI_IMAGE_GENERATOR_PRESENT ) )
    {
        cout <<
            "\nImage generator output mode:" << endl <<
            "FRAME_WIDTH   " << capture1.get( CV_CAP_OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_FRAME_WIDTH ) << 
            "   | FRAME_HEIGHT  " << capture1.get( CV_CAP_OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_FRAME_HEIGHT ) << 
            "   | FPS           " << capture1.get( CV_CAP_OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_FPS ) << endl;
    }
    else
    {
        cout << "\nDevice doesn't contain image generator." << endl;
    }

    int success = 0, k = 0;
    bool rgb_done = false;
    bool depth_done = true;
    bool found1 = false, found2 = false;

    while (rgb_done != true)
    {
        if( !capture1.grab() )
        {
            cout << "Can not grab images." << endl;
            return -1;
        }
            //capture.retrieve( depth, CV_CAP_OPENNI_DEPTH_MAP );
            
            capture1.retrieve( img1, CV_CAP_OPENNI_BGR_IMAGE );
        cvtColor(img1, gray1, CV_BGR2GRAY);

        found1 = findChessboardCorners(img1, board_sz, corners1, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        //found2 = findChessboardCorners(img2, board_sz, corners2, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

        if (found1)
        {
            cornerSubPix(gray1, corners1, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(gray1, board_sz, corners1, found1);
        }

//        if (found2)
//        {
//            cornerSubPix(gray2, corners2, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
//            drawChessboardCorners(gray2, board_sz, corners2, found2);
//        }
        
        imshow("image1", gray1);
       // imshow("image2", gray2);

        k = waitKey(10);
        if (found1)
        {
            k = waitKey(0);
        }
        if (k == 27)
        {
            break;
        }
        if (k == ' ' && found1 !=0)
        {
            imagePoints1.push_back(corners1);
            //imagePoints2.push_back(corners2);
            object_points.push_back(obj);
            printf ("Corners stored\n");
            success++;
            rgb_done = true;
           
        }
    }

    
    VideoCapture capture2;
    capture2.set(CV_CAP_PROP_FOURCC, CV_FOURCC('D','I','V','4'));
    capture2.open( "ir_pics/ir-stereo.png" );
    
    if (!capture2.isOpened()){
    cout << "error loading file" << endl;
    return -1;
    }
    
    if( !capture2.grab() )
        {
            cout << "Can not grab images." << endl;
            return -1;
        }
            //capture.retrieve( depth, CV_CAP_OPENNI_DEPTH_MAP );
            
            capture2.retrieve(img2);
            cvtColor(img2, gray2, CV_BGR2GRAY);
            
            found2 = findChessboardCorners(img2, board_sz, corners2, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
    
            if (found2)
        {
            cornerSubPix(gray2, corners2, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(gray2, board_sz, corners2, found2);
        }
            imshow("image2", gray2);
            
            k = waitKey(10);
        if (found2)
        {
            cout << "Depth Image valid" << endl;
            k = waitKey(0);
        }
        if (!found2)
        {
            cout << "Depth Image is not valid. Change Image and try again. Aborting !!!" << endl;
            return -1;
        }
        if (k == ' ' && found1 !=0)
        {
            //imagePoints1.push_back(corners1);
            imagePoints2.push_back(corners2);
           // object_points.push_back(obj);
            printf ("Corners stored\n");
           // success++;
            depth_done = true;
           
        }
            
    if(!(rgb_done && depth_done)){
        cout << "Error in storing" << endl;
        return -1;
    }       
            
    destroyAllWindows();
    printf("Starting Calibration\n");
    Mat CM1,CM2,D1,D2,R, T, E, F;
    FileStorage fs1(rgb_intrinsics, FileStorage::READ);
    FileStorage fs2(depth_intrinsics, FileStorage::READ);
    fs1["CM1"] >> CM1;
    fs2["CM1"] >> CM2;
    //Mat CM1 = Mat(3, 3, CV_64FC1);
    //Mat CM2 = Mat(3, 3, CV_64FC1);
     fs1["D1"] >> D1;
     fs1["D1"] >> D2;
    //Mat CM1,CM2,D1,D2,R, T, E, F;

    stereoCalibrate(object_points, imagePoints1, imagePoints2, 
                    CM1, D1, CM2, D2, img1.size(), R, T, E, F,
                    cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5), 
                    CV_CALIB_FIX_INTRINSIC);

    FileStorage fs3("mystereocalib1.yml", FileStorage::WRITE);
    fs3 << "CM1" << CM1;
    fs3 << "CM2" << CM2;
    fs3 << "D1" << D1;
    fs3 << "D2" << D2;
    fs3 << "R" << R;
    fs3 << "T" << T;
    fs3 << "E" << E;
    fs3 << "F" << F;

    printf("Done Calibration\n");

    printf("Starting Rectification\n");

    Mat R1, R2, P1, P2, Q;
    stereoRectify(CM1, D1, CM2, D2, img1.size(), R, T, R1, R2, P1, P2, Q);
    fs3 << "R1" << R1;
    fs3 << "R2" << R2;
    fs3 << "P1" << P1;
    fs3 << "P2" << P2;
    fs3 << "Q" << Q;

    printf("Done Rectification\n");

//    printf("Applying Undistort\n");
//
//    Mat map1x, map1y, map2x, map2y;
//    Mat imgU1, imgU2;
//
//    initUndistortRectifyMap(CM1, D1, R1, P1, img1.size(), CV_32FC1, map1x, map1y);
//    initUndistortRectifyMap(CM2, D2, R2, P2, img2.size(), CV_32FC1, map2x, map2y);
//
//    printf("Undistort complete\n");
//
//    while(1)
//    {    
//        cap1 >> img1;
//        cap2 >> img2;
//
//        remap(img1, imgU1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
//        remap(img2, imgU2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
//
//        imshow("image1", imgU1);
//        imshow("image2", imgU2);
//
//        k = waitKey(5);
//
//        if(k==27)
//        {
//            break;
//        }
//    }
//
//    cap1.release();
//    cap2.release();

    return(0);
}
