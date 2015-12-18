#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <signal.h>

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <chrono>
#include <string.h>
#include <sstream>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <opencv/ml.h>
//#include <iostream>
//#include <stdio.h>
#include <cctype> 
#include <opencv2/opencv.hpp>
//#include <opencv2/gpu/gpu.hpp>
//#include "opencv2/core/utility.hpp"
//#include "opencv/core/ocl.hpp"

using namespace cv;
using namespace std;
//using namespace cv::gpu;

/** Global variables */

const String face_cascade_name = "haarcascade_frontalface_alt2.xml";
const String eyes_cascade_name = "haarcascade_eye.xml";

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

//CascadeClassifier face_cascade_gpu;
//CascadeClassifier eyes_cascade_gpu;

string window_name = "Xbox features viewer";
RNG rng(12345);

//font properties
static const cv::Point pos(5, 15);
static const cv::Scalar colorText = CV_RGB(0, 0, 255);
static const double sizeText = 0.5;
static const int lineText = 1;
static const int font = cv::FONT_HERSHEY_SIMPLEX;

KalmanFilter KF(2, 1, 0);
Mat processNoise(2, 1, CV_32F);
Mat state(2, 1, CV_32F);

const float Qt = 2000; //1500.0;
const float Rt =  100;//70;//52.6832;          //hardcoding variance of xbox sensor

float xboxdepth;          //at global scope
float xboxdepth1;
float worldx;
float worldy;
float imagex;
float imagey;
/** Function Prototypes */
Mat detectfeatures( Mat color, Mat depthMap );
void talker(float& xboxobs, Mat prediction, Mat update, Mat Pkkm1, Mat Pkk, Mat Kk);
void kalman(float deltaT, Mat measurement);

//Mat depth, color;

// Camera Calibration Parameters

//Intrinsic
//const Matx33f f(5.2334888056605109e+02, 0, 3.3077859778899148e+02, 0,
//       5.2640349339321790e+02, 2.4796327924961466e+02, 0, 0, 1);
//
//const Matx33f f_d(5.9421434211923247e+02,0,3.3930780975300314e+02,
//        0,5.9104053696870778e+02,2.4273913761751615e+02,0,0,1);
////Distortion
//Mat_<float> dstCoeff(5,1);
//Mat_<float> dstCoeff_d(5,1);
//
//
// 
////Extrinsic
//const Matx34f RT( 9.9984628826577793e-01, 1.2635359098409581e-03,
//-1.7487233004436643e-02, -1.4779096108364480e-03,
//9.9992385683542895e-01, -1.2251380107679535e-02,
//1.7470421412464927e-02, 1.2275341476520762e-02,
//9.9977202419716948e-01 );


//Mat I = Mat(f);
//Mat E = Mat(RT);

//Mat P = I*E;
//Matx34f Ptemp = f*RT;
//Matx33f Prod1(Ptemp(0,0),Ptemp(0,1),Ptemp(0,2),
               // Ptemp(1,0),Ptemp(1,1),Ptemp(0,2),
               // Ptemp(2,0),Ptemp(2,1),Ptemp(2,2));

//Distortion
//const float dist =  1.2453643444153988e-01; // will change. Using this for calculation purpose.
       
//Matx33f Prod = Prod1; //first 3 columns of projection matrix

//Matx31f T(Ptemp(0,3),Ptemp(1,3),Ptemp(2,3)); // last column of projection matrix
       //unsigned char *offset = (unsigned char*)(T.data);
      // float offx = float(&offset[0]);
      // float offy = float(&offset[1]);
      // float offz = float(&offset[2]);
  Mat CM1,CM2,D1,D2,R, T,RotT,P,P1,RotT1;
  
static void help()
{
        cout << "\nAll supported output map types:\n"
            "1.) Data given from depth generator\n"
            "   OPENNI_DEPTH_MAP            - depth values in mm (CV_16UC1)\n"
            "   OPENNI_POINT_CLOUD_MAP      - XYZ in meters (CV_32FC3)\n"
            "   OPENNI_DISPARITY_MAP        - disparity in pixels (CV_8UC1)\n"
            "   OPENNI_DISPARITY_MAP_32F    - disparity in pixels (CV_32FC1)\n"
            "   OPENNI_VALID_DEPTH_MASK     - mask of valid pixels (not ocluded, not shaded etc.) (CV_8UC1)\n"
            "2.) Data given from RGB image generator\n"
            "   OPENNI_BGR_IMAGE            - color image (CV_8UC3)\n"
            "   OPENNI_GRAY_IMAGE           - gray image (CV_8UC1)\n"
         << endl;
}

static void printCommandLineParams()
{
    cout << "-cd       Colorized disparity? (0 or 1; 1 by default) Ignored if disparity map is not selected to show." << endl;
    cout << "-mode     image mode: resolution and fps, supported three values:  0 - CV_CAP_OPENNI_VGA_30HZ, 1 - CV_CAP_OPENNI_SXGA_15HZ," << endl;
    cout << "-s        Save depth points to yamlfile? (0 or 1; 0 by default) Ignored if -cd 0 is not included as command line terminal" << endl;
}


static void parseCommandLine( int argc, char* argv[], int& imageMode, bool& save)
{
    // set defaut values   
    save = false;
    imageMode = 0;

    if( argc == 1 )
    {
        help();
    }
    else
    {
        for( int i = 1; i < argc; i++ )
        {
            if( !strcmp( argv[i], "--help" ) || !strcmp( argv[i], "-h" ) )
            {
                printCommandLineParams();
                exit(0);
            }
            else if( !strcmp( argv[i], "-mode" ) )
            {
                imageMode = atoi(argv[++i]);
            }
            else if( !strcmp( argv[i], "-s") )
            {
                save = atoi(argv[++i]) == 0 ? false : true;
            }
            else
            {
                cout << "Unsupported command line argument: " << argv[i] << "." << endl;
                exit(-1);
            }
        }
    }
}

const String observations = "Xbox_obs.yaml";
const String predictions  = "Xbox_Pred.yaml";
const String estimates    = "Xbox_Updates.yaml";

vector<float> updwin;
Mat  HT, inter,   Rk;

  void kalman(float deltaT, Mat measurement)
  {
    //Make Q(k) a random walk

    float q11 = pow(deltaT, 4)/4.0 ;
    float q12 = pow(deltaT, 3)/2.0 ;
    float q21 = pow(deltaT, 3)/2.0 ;
    float q22 = pow(deltaT, 2) ;

    KF.processNoiseCov = *(Mat_<float>(2,2) << q11, q12, q21, q22);

    KF.processNoiseCov *=Qt;

    KF.transitionMatrix = *(Mat_<float>(2, 2) << 1, deltaT, 0, 1);
    
    Mat prediction = KF.predict(); 

    Mat update =  KF.correct(measurement); 

    Mat Pkk = KF.errorCovPost;
    Mat Pkkm1 = KF.errorCovPre;
    Mat Kk = KF.gain;

    //cout << "gain matrix: " << KF.gain << endl;

    float xboxpred = prediction.at<float>(0);
    float xboxupd  = update.at<float>(0);
    float xboxobs = measurement.at<float>(0); 

    //talk values in a named pipe
    talker(xboxobs, prediction, update, Pkkm1, Pkk, Kk) ;     
/*
    cv::FileStorage fx;
    fx.open(estimates, cv::FileStorage::APPEND);
    fx << "estimates" << xboxupd;
    fx.release();

    cv::FileStorage fy;
    fy.open(predictions, cv::FileStorage::APPEND);
    fy << "prediction" << xboxpred;
    fy.release();

    cv::FileStorage fz;
    fz.open(observations, cv::FileStorage::APPEND);
    fz << "observation" << xboxobs;
    fz.release(); 
    */
  }
    /* Catch Signal Handler function */
    void signal_callback_handler(int signum)
    {
        printf("Caught signal SIGPIPE: %d\n",signum);
    }

void talker(float& xboxobs, Mat prediction, Mat update, Mat Pkkm1, Mat Pkk, Mat Kk)
    {
        int fm, fp, fp1, fu, fu1; 
        int fpe, fpe1, fpe2, fpe3;
        int fee, fee1, fee2, fee3;
        int fg, fg1;    

        int mat;   


        float xboxpred    = prediction.at<float>(0);
        float xboxpred1   = prediction.at<float>(1);
        float xboxupd     = update.at<float>(0);
        float xboxupd1    = update.at<float>(1);
        float gain        = KF.gain.at<float>(0);
        float gain1       = KF.gain.at<float>(1);

        float pred_error  = Pkkm1.at<float>(0, 0);
        float pred_error1 = Pkkm1.at<float>(0, 1);
        float pred_error2 = Pkkm1.at<float>(1, 0);
        float pred_error3 = Pkkm1.at<float>(1, 1);

        float est_error   = Pkk.at<float>(0, 0);
        float est_error1  = Pkk.at<float>(0, 1);
        float est_error2  = Pkk.at<float>(1, 0);
        float est_error3  = Pkk.at<float>(1, 1);


        //Measurement FIFO
        const char * obsfifo = "/tmp/obsfifo";
        mkfifo(obsfifo, 0666);                       
        fm = open(obsfifo, O_WRONLY | O_NONBLOCK);         
        write(fm, &xboxobs, sizeof(xboxobs) ); 
        close(fm);       

        //Kalman Prediction FIFO
        const char * predfifo = "/tmp/predfifo";
        mkfifo(predfifo, 0666);                       
        fp = open(predfifo, O_WRONLY | O_NONBLOCK);          
        write(fp, &xboxpred, sizeof(xboxpred) ); 
        close(fp);    

        const char * predfifo1 = "/tmp/predfifo1";
        mkfifo(predfifo1, 0666);                       
        fp1 = open(predfifo1, O_WRONLY | O_NONBLOCK);          
        write(fp1, &xboxpred1, sizeof(xboxpred1) ); 
        close(fp1);

        //Kalman Update FIFO
        const char * updfifo = "/tmp/updfifo";
        mkfifo(updfifo, 0666);                       
        fu = open(updfifo, O_WRONLY | O_NONBLOCK);           
        write(fu, &xboxupd, sizeof(xboxupd) );   
        close(fu);

        const char * updfifo1 = "/tmp/updfifo1";
        mkfifo(updfifo1, 0666);                       
        fu1 = open(updfifo1, O_WRONLY | O_NONBLOCK);           
        write(fu1, &xboxupd1, sizeof(xboxupd1) );   
        close(fu1);

        //Kalman Prediction error FIFO
        const char * prederrorfifo = "/tmp/prederrorfifo";
        mkfifo(prederrorfifo, 0666);                       
        fpe = open(prederrorfifo, O_WRONLY | O_NONBLOCK);           
        write(fpe, &pred_error, sizeof(pred_error) );   
        close(fpe);

        const char * prederrorfifo1 = "/tmp/prederrorfifo1";
        mkfifo(prederrorfifo1, 0666);                       
        fpe1 = open(prederrorfifo1, O_WRONLY | O_NONBLOCK);           
        write(fpe1, &pred_error1, sizeof(pred_error1) );   
        close(fpe1);

        const char * prederrorfifo2 = "/tmp/prederrorfifo2";
        mkfifo(prederrorfifo2, 0666);                       
        fpe2 = open(prederrorfifo2, O_WRONLY | O_NONBLOCK);           
        write(fpe2, &pred_error2, sizeof(pred_error2) );   
        close(fpe2);

        const char * prederrorfifo3 = "/tmp/prederrorfifo3";
        mkfifo(prederrorfifo3, 0666);                       
        fpe3 = open(prederrorfifo3, O_WRONLY | O_NONBLOCK);           
        write(fpe3, &pred_error3, sizeof(pred_error3) );   
        close(fpe3);

        //Kalman Estimation error FIFO
        const char * esterrorfifo = "/tmp/esterrorfifo";
        mkfifo(esterrorfifo, 0666);                       
        fee = open(esterrorfifo, O_WRONLY | O_NONBLOCK);           
        write(fee, &est_error, sizeof(est_error) );   
        close(fee);

        const char * esterrorfifo1 = "/tmp/esterrorfifo1";
        mkfifo(esterrorfifo1, 0666);                       
        fee1 = open(esterrorfifo1, O_WRONLY | O_NONBLOCK);           
        write(fee1, &est_error1, sizeof(est_error1) );   
        close(fee1);

        const char * esterrorfifo2 = "/tmp/esterrorfifo2";
        mkfifo(esterrorfifo2, 0666);                       
        fee2 = open(esterrorfifo2, O_WRONLY | O_NONBLOCK);           
        write(fee2, &est_error2, sizeof(est_error2) );   
        close(fee2);

        const char * esterrorfifo3 = "/tmp/esterrorfifo3";
        mkfifo(esterrorfifo3, 0666);                       
        fee3 = open(esterrorfifo3, O_WRONLY | O_NONBLOCK);           
        write(fee3, &est_error3, sizeof(est_error3) );   
        close(fee3);

        //Kalman gain  FIFO
        const char * gainfifo = "/tmp/gainfifo";
        mkfifo(gainfifo, 0666);                       
        fg = open(gainfifo, O_WRONLY | O_NONBLOCK);           
        write(fg, &gain, sizeof(gain) );   
        close(fg);

        const char * gainfifo1 = "/tmp/gainfifo1";
        mkfifo(gainfifo1, 0666);                       
        fg1 = open(gainfifo1, O_WRONLY | O_NONBLOCK);           
        write(fg1, &gain1, sizeof(gain1) );   
        close(fg1);

        cout << "   | xboxobs: " << xboxobs <<
                "   | xboxpred: " << xboxpred <<
                "   | update: " << xboxupd<< endl;   

        cout << "est_error: " << est_error << " | pred_error: " << pred_error << " | gain: "<< gain << endl;
    }

    size_t frameCount = 0;
    double fps = 0;
    double totalT = 0.0;
    long frmCnt = 0;
    double elapsed;
    Size dsize;
    double fx = 0.95;
    double fy = 0.95;
    int interpol=INTER_LINEAR;
    

    const double scaleFactor = 1.2;
    const int minNeighbors = 6;

    const Size face_maxSize = Size(20, 20);
    const Size face_minSize = Size(5, 5);

    std::ostringstream oss;
    
    float raw_depth_to_meters(short raw_depth) // Kinect depth translation. raw to depth in meters.
{
  if (raw_depth < 2047)
  {
   return 1.0 / (raw_depth * -0.0030711016 + 3.3309495161);
  }
  return 0;
}
//    Point_<float> compute_world_points(float imagex, float imagey, float xboxdepth1){
//  ////////////////////Computation begins here ////////////////////////////////////////////      
//        
//      Matx33f IP(imagex - T(0,0), 
//                    imagey- T(0,1),
//                     1 - T(0,2));
//
//      //Matx33f WP = Prod.inv()*IP;
//      // cv::solve(Prod,IP,WP);
//       
//       cout<< "ROWS"<<IP.rows << endl;
//       cout<< "columns"<<IP.cols << endl;
//       //cout<< RT(2,2)<<endl;
//       // cout << WP(0,0) << endl;
//       // cout << WP(1,0)<< endl;
//       // cout << WP(2,0) << endl;
//      // unsigned char *wp = (unsigned char*)(IP.data);
//       
//      worldx = float(((WP(0,0)/WP(2,0))*xboxdepth1));
//      worldy = float(((WP(1,0)/WP(2,0))*xboxdepth1));
//       Point_<float> worldxy(worldx,worldy);
//       
//       
//       return worldxy;
//    
//    }
    
    Point_<double> compute_world_points1(double imagex, double imagey, Mat depth){
//        Point_<float> Depth3D;
        Mat ip = (Mat_<double>(3,1) << (320-imagex),(imagey-240),1);
        Mat ip_new = CM1.inv()*ip;
        Mat wp = (ip_new-T);
        Mat wp_new = R.t()*wp; 
//        float ip[4][1] = {imagex,imagey, 1,1};
//        Mat ImagePoints = Mat(4,1,CV_64F, ip);
//        float wp[4][1] = {1,1,1,1};
//        Mat WorldPoints = Mat(4,1,CV_64F,wp);
//        WorldPoints = P.inv()*ImagePoints;
        //cout <<  "World matrix" << ip_new << endl;
        //scout <<  "World matrix" << wp_new << endl;
         //cout <<  "World matrix" << wp.at<float>(2,0) << endl;
        // cout <<  "World matrix" << WorldPoints.at<float>(3,0) << endl;
        // add row for CM1 her
         Point_<double> worldxy;
         
         worldxy.x = (wp.at<double>(0,0)/wp.at<double>(2,0))*raw_depth_to_meters((depth.at<unsigned short>(imagey, imagex) - 208 - 175));
         worldxy.y = (wp.at<double>(1,0)/wp.at<double>(2,0))*raw_depth_to_meters((depth.at<unsigned short>(imagey, imagex) - 208 - 175));
         return worldxy;
  
    }
    
//   void addvalues_dstCoeff(){
//        dstCoeff.at<float>(0,0) = 1.8214054882125558e-01;
//        dstCoeff.at<float>(1,0) = -5.6454045166848910e-01;
//        dstCoeff.at<float>(2,0) = -1.3961377280662568e-02;
//        dstCoeff.at<float>(3,0) = 5.0548940995638501e-03;
//        dstCoeff.at<float>(4,0) = 1.0268389479216953e+00;
//        
//        dstCoeff_d.at<float>(0,0) = -2.6386489753128833e-01;
//        dstCoeff_d.at<float>(1,0) = 9.9966832163729757e-01;
//        dstCoeff_d.at<float>(2,0) = -7.6275862143610667e-04;
//        dstCoeff_d.at<float>(3,0) = 5.0350940090814270e-03;
//        dstCoeff_d.at<float>(4,0) = -1.3053628089976321e+00;
//   
//    }
    
    
Mat detectfeatures(Mat color, Mat depth)
{
    Mat frame_gray, gray_resized, color_resized, depth_resized;
    Mat faces_host, eyes_host;
    //Mat eyes,faces;
   
    double t = (double)getTickCount();

    cvtColor( color, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
        
    resize(frame_gray, gray_resized, Size(), fx, fy, interpol);
    resize(color, color_resized, Size(), fx, fy, interpol);         //for opencv window       
    resize(depth, depth_resized, Size(), fx, fy, interpol);           
    //Mat frame_gray_gpu(gray_resized);  
     Mat R,mapx,mapy,fnew,R1,mapx_d,mapy_d,f_dnew;
     
     //move grayed color image to gpu
    initUndistortRectifyMap(CM1,D1,R,fnew,Size(fx*640,fy*480),CV_32FC1,mapx,mapy);
    initUndistortRectifyMap(CM2,D2,R1,f_dnew,Size(fx*640,fy*480),CV_32FC1,mapx_d,mapy_d);
    remap(gray_resized,gray_resized,mapx,mapy,INTER_LINEAR);
    remap(color_resized,color_resized,mapx,mapy,INTER_LINEAR);
    remap(depth_resized,depth_resized,mapx_d,mapy_d,INTER_LINEAR);
    //-- Detect faces    
    //faces.create(1, 100, cv::DataType<cv::Rect>::type);   //preallocate gpu faces
     std::vector<Rect> faces;
    
   // int faces_detect = face_cascade.detectMulti //-- Detect faces
  face_cascade.detectMultiScale( gray_resized, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

  
  t = ( (double)getTickCount() - t)/getTickFrequency();               //measure total time to detect and download

    totalT += t;
    ++frmCnt;

    cout << "\nfps: " << 1.0 / (totalT / (double)frmCnt) << endl;
    
  for( size_t i = 0; i < faces.size(); i++ )
  {
      std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
              
        start = chrono::high_resolution_clock::now();
      
    Point face_center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    
    ellipse( color_resized, face_center, Size( faces[i].width*0.5, faces[i].height*0.75), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

    Mat faceROI = gray_resized( faces[i] );
    std::vector<Rect> eyes;

    //-- In each face, detect eyes
    eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    for( size_t j = 0; j < eyes.size(); j++ )
     {
       Point eye_center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
       int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
       circle( color_resized, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
       if(eye_center.x > face_center.x)
       xboxdepth  = raw_depth_to_meters((depth_resized.at<unsigned short>(eye_center.y, eye_center.x) - 208 - 175));
       else if (eye_center.x < face_center.x){
       xboxdepth1 = raw_depth_to_meters((depth_resized.at<unsigned short>(eye_center.y, eye_center.x) - 208 - 175));
       imagex = double(eye_center.x);
       imagey = double(eye_center.y);
       //Matx41f scalar(1,1,1,1);
       // S = Mat(scalar);
       
       }
     }
    
        

        end = chrono::high_resolution_clock::now(); 
        
        float deltaT = chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
        Mat measurement = Mat(1, 1, CV_32F, xboxdepth);
        
        Point_<float> worldxy = compute_world_points1(imagex,imagey,depth_resized);
        
       // cout << worldxy.x << endl;
       // cout << worldxy.y << endl;
        
        std::ostringstream osf; 
        std::ostringstream osf1;
        std::ostringstream osf2;
        osf.str(" ");
        osf <<"World Z " << xboxdepth1 << " m";
        
        osf1.str(" ");
        osf1 <<"World X " << worldxy.x*1000 << " mm";
        
        osf2.str(" ");
        osf2 <<"World Y " << worldxy.y*1000 << " mm";
        
        putText(color, oss.str(), Point(20,35), font, sizeText, colorText, lineText,CV_AA);
        putText(color_resized, osf1.str(), Point(20,55), font, sizeText, colorText, lineText,CV_AA);
        putText(color_resized, osf2.str(), Point(20,75), font, sizeText, colorText, lineText,CV_AA); 
        putText(color_resized, osf.str(), Point(20,95), font, sizeText, colorText, lineText,CV_AA);
        
        cout <<  "\n\n deltaT: " << deltaT << endl;
        kalman(deltaT, measurement);       //filter observation from kinect    
    }

    return color_resized;  
}


int main( int argc, char* argv[] )
{
    bool isColorizeDisp, isFixedMaxDisp;
    int imageMode;

    string filename;
    bool save, isVideoReading;
    parseCommandLine( argc, argv, imageMode, save );

    //load cpu cascades
    face_cascade.load( face_cascade_name );
    eyes_cascade.load( eyes_cascade_name );

    //load gpu cascades
    //face_cascade_gpu.load( face_cascade_name );
    //eyes_cascade_gpu.load( eyes_cascade_name );

    //int gpuCnt = getCudaEnabledDeviceCount();   // gpuCnt >0 if CUDA device detected

    //cout <<"Number of Gpus: " << gpuCnt << endl;

    //if(gpuCnt==0)
   // {
    //    return 0;                       // no CUDA device found, quit
    //} 

    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(Qt));
    setIdentity(KF.measurementNoiseCov, Scalar::all(Rt));
    setIdentity(KF.errorCovPost, Scalar::all(1));

    KF.statePost.at<float>(0) = 687;

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
    
    /* Catch Signal Handler SIGPIPE */
    signal(SIGPIPE, signal_callback_handler);
    
    std::chrono::time_point<std::chrono::high_resolution_clock> begin, now,time_seconds;    
    std::ostringstream oss;
    oss << "starting...";

    begin = std::chrono::high_resolution_clock::now();
    cout << "\nOpening files" << endl;
    //addvalues_dstCoeff();
    FileStorage fs1("mystereocalib1.yml", FileStorage::READ);
    fs1["CM1"] >> CM1;
    fs1["CM2"] >> CM2;
    //Mat CM1 = Mat(3, 3, CV_64FC1);
    //Mat CM2 = Mat(3, 3, CV_64FC1);
     fs1["D1"] >> D1;
     fs1["D2"] >> D2;
     fs1["R"]  >> R;
     fs1["T"]  >> T;
     fs1.release();
      cout << "\nOpening files" << endl;
     // Mat RotT(3,3,CV_32FC1);
     cout << "\nOpening files" << endl;
     //R.copyTo(RotT);
     //T.copyTo(Mat(RotT.col(3)));
     hconcat(R,T,RotT);
     //assert(3 == RotT.rows && 4 == RotT.cols);
     
     cout << "\nOpening file5" << endl;
     P = CM1*RotT;
     cout << "\nOpening file5"  << endl;
     Mat dummy = (Mat_<double>(1,4) << 0,0,0,1);
      cout << "\nOpening files" << P <<  endl;
    // P.row(3) = Mat(1,4,CV_32F,dummy);
      //dummy.copyTo(P.row(3));
      //Mat P0[] = {P,dummy};
      vconcat(P,dummy,P1);
     // assert(4 == P1.rows && 4 == P1.cols);
    cout << "\nOpening files2" << endl;
    
    for(;;)
    {  
        //time_seconds = std::chrono::high_resolution_clock::now();
        Mat color, depth;
        if( !capture.grab() )
        {
            cout << "Can not grab images." << endl;
            return -1;
        }
            capture.retrieve( depth, CV_CAP_OPENNI_DEPTH_MAP );
            
            capture.retrieve( color, CV_CAP_OPENNI_BGR_IMAGE );
//            if ((std::chrono::duration_cast<std::chrono::milliseconds>(time_seconds-begin).count() / 1000.0) > 1){
//            cout << "Entering face Detect" << endl;
            Mat color_resized = detectfeatures(color, depth);
            imshow( "Xbox Features Viewer", color_resized );
//            begin = time_seconds;
//            }
//            else{
//                imshow( "Xbox Features Viewer", color);
//            }
            now = std::chrono::high_resolution_clock::now();        
            ++frameCount;
            elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - begin).count() / 1000.0;
       
        if(elapsed >= 1.0)
        {
        fps = frameCount / elapsed;
        oss << "fps: " << fps << " ( " << elapsed / frameCount * 1000.0 << " ms)";
        begin = now;
        frameCount = 0;
        }
        if( waitKey( 30 ) >= 0 )
        break;
    }
    return 0;
}


/*
cd ../; rm -rf build; mkdir build; cd build; cmake ../; make; cp ../Haar/*.xml `pwd`; ./xbox_tracker
*/
