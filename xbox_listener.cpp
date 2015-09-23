#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>

#include <string.h>
#include <iostream>
#include <string>
#include <sstream>

#include <chrono>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <boost/asio.hpp>
#include "boost/bind.hpp"
#include "boost/date_time/posix_time/posix_time_types.hpp"

using namespace std;
using namespace Eigen;
using namespace cv;

const short multicast_port = 30001;
const int max_message_count = 1;

float easytracktotrack(Vector2f Xbox_update, /*Vector2f ROS_update*/ Matrix2f Xbox_Pkk /*Matrix2f ROS_Pkk*/);

class sender
{
public:
  sender(boost::asio::io_service& io_service,
      const boost::asio::ip::address& multicast_address, float fusedpos_)
    : endpoint_(multicast_address, multicast_port),
      socket_(io_service, endpoint_.protocol()),
      timer_(io_service)
  {
    fusion = fusedpos_;
    //cout << "sender fusion: " << fusion << endl;
    std::ostringstream os;
    os << fusion;
    message_ = os.str();

    socket_.async_send_to(
        boost::asio::buffer(message_), endpoint_,
        boost::bind(&sender::handle_send_to, this,
          boost::asio::placeholders::error));
  }

  void handle_send_to(const boost::system::error_code& error)
  {
    if (!error && fusion< max_message_count)
    {
      timer_.expires_from_now(boost::posix_time::seconds(1));
      timer_.async_wait(
          boost::bind(&sender::handle_timeout, this,
            boost::asio::placeholders::error));
    }
  }

  void handle_timeout(const boost::system::error_code& error)
  {
    if (!error)
    {
      std::ostringstream os;
      os << "Fusion: " << fusion;
      message_ = os.str();

      socket_.async_send_to(
          boost::asio::buffer(message_), endpoint_,
          boost::bind(&sender::handle_send_to, this,
            boost::asio::placeholders::error));
    }
  }

private:
  boost::asio::ip::udp::endpoint endpoint_;
  boost::asio::ip::udp::socket socket_;
  boost::asio::deadline_timer timer_;
  float fusion;
  std::string message_;

};

float easytracktotrack(Vector2f Xbox_update, /*Vector2f ROS_update,*/ Matrix2f Xbox_Pkk /*Matrix2f ROS_Pkk*/)
{
   /*Compute P_T(k|k)*/
   Matrix2f P_Tkk =  Xbox_Pkk.inverse() /*+ ROS_Pkk.inverse() ).inverse()*/;

   /*fusion internals*/
   Vector2f term1 = Xbox_Pkk.inverse() * Xbox_update;
   //Vector2f term2 = ROS_Pkk.inverse()  * ROS_update;
   Vector2f term3 = term1;// + term2;

   /*Straight track to track*/
   Vector2f fusion = P_Tkk * term3; 

   //cout << "\n\nStraight Fusion: [" << fusion.transpose() << "]" << endl;

   float fusedpos_ = fusion(0);

   /*Write what you got*/
   /* float& xboxval = Xbox_update.at<float>(0);
   cv::FileStorage fa;
   const String xbox_estimates = "Xbox_estimates.yaml";
   fa.open(xbox_estimates, cv::FileStorage::APPEND);
   fa << "Xbox_estimates" << xboxval;
   fa.release();
   float& rosval = ROS_update.at<float>(0);
   cv::FileStorage fb;
   const String ros_estimates = "ros_estimates.yaml";
   fb.open(ros_estimates, cv::FileStorage::APPEND);
   fb << "ros_estimates" << rosval;
   fb.release();
   float& fused = fusion.at<float>(0);
   cv::FileStorage fc;
   const String easy_fusion = "easy_fusion.yaml";
   fc.open(easy_fusion, cv::FileStorage::APPEND);
   fc << "easy_fusion" << fused;
   fc.release();*/

   return fusedpos_;
}

int main(int argc, char* argv[])
{ 
    /*ROS Pipes*/
   // int rosfm, rosfp, rosfp1, rosfu, rosfu1;                    //obs, predict, estimate fifos
   // int rosfpe, rosfpe1, rosfpe2, rosfpe3;                      //ros prediction error fifo
   // int rosfee, rosfee1, rosfee2, rosfee3;                      //ros estimate error fifo

    /*Xbox Pipes*/
    int fp, fp1, fu, fu1; 
    int fpe, fpe1, fpe2, fpe3;
    int fee, fee1, fee2, fee3;


    //Xbox Prediction fifos
    char * predfifo       = "/tmp/predfifo";
    char * predfifo1      = "/tmp/predfifo1";
    //Xbox estimation fifos
    char * updfifo        = "/tmp/updfifo";
    char * updfifo1       = "/tmp/updfifo1";
    //Xbox Prediction error fifos
    char * prederrorfifo  = "/tmp/prederrorfifo";
    char * prederrorfifo1 = "/tmp/prederrorfifo1";
    char * prederrorfifo2 = "/tmp/prederrorfifo2";
    char * prederrorfifo3 = "/tmp/prederrorfifo3";
    //Xbox Estimation error fifos
    char * esterrorfifo   = "/tmp/esterrorfifo";
    char * esterrorfifo1  = "/tmp/esterrorfifo1";
    char * esterrorfifo2  = "/tmp/esterrorfifo2";
    char * esterrorfifo3  = "/tmp/esterrorfifo3";

    //Kalman Prediction FIFO
//    char * rospredfifo    = "/tmp/rospredfifo";
//    char * rospredfifo1   = "/tmp/rospredfifo1";
//    //Kalman Update FIFO
//    char * rosupdfifo     = "/tmp/rosupdfifo";
//    char * rosupdfifo1    = "/tmp/rosupdfifo1";
  //Kalman gain  FIFO

    //Kalman Prediction error FIFO
//    char * rosprederrorfifo = "/tmp/rosprederrorfifo";
//    char * rosprederrorfifo1 = "/tmp/rosprederrorfifo1";
//    char * rosprederrorfifo2 = "/tmp/rosprederrorfifo2";
//    char * rosprederrorfifo3 = "/tmp/rosprederrorfifo3";
//    //Kalman Estimation error FIFO
//    char * rosesterrorfifo = "/tmp/rosesterrorfifo";
//    char * rosesterrorfifo1 = "/tmp/rosesterrorfifo1";
//    char * rosesterrorfifo2 = "/tmp/rosesterrorfifo2";
//    char * rosesterrorfifo3 = "/tmp/rosesterrorfifo3";

    std::chrono::time_point<std::chrono::high_resolution_clock> begin, now; 
    while(1)
    {
        begin = std::chrono::high_resolution_clock::now();
        //READ kalman measurement fifo

        //READ kalman prediction fifo
        fp = open(predfifo, O_RDONLY  );
        float pred;
        read(fp, &pred, sizeof(pred) );
        close(fp);
        //printf("    | pred: %4.2f", pred); 
        fflush(stdout);

        fp1 = open(predfifo1, O_RDONLY  );
        float pred1;
        read(fp1, &pred1, sizeof(pred1) );
        close(fp1);
        //printf("    | pred1: %4.2f", pred1); 
        fflush(stdout);
    
        //READ kalman estimate fifo
        fu = open(updfifo, O_RDONLY  );
        float upd  ;
        read(fu, &upd, sizeof(upd));
        close(fu);
        //printf("    | update: %4.2f", upd);  
        fflush(stdout);

        fu1 = open(updfifo1, O_RDONLY  );
        float upd1  ;
        read(fu1, &upd1, sizeof(upd1));
        close(fu1);
        //printf("    | update1: %4.2f", upd1);  
        fflush(stdout);

        //READ kalman prediction error fifo
        fpe = open(prederrorfifo, O_RDONLY  );
        float pred_error  ;
        read(fpe, &pred_error, sizeof(pred_error));
        close(fpe);
        //printf("    | errorPred: %4.2f", pred_error);  
        fflush(stdout);

        fpe1 = open(prederrorfifo1, O_RDONLY  );
        float pred_error1  ;
        read(fpe1, &pred_error1, sizeof(pred_error1));
        close(fpe1);
        //printf("    | errorPred:1 %4.2f", pred_error1);  
        fflush(stdout);

        fpe2 = open(prederrorfifo2, O_RDONLY  );
        float pred_error2  ;
        read(fpe2, &pred_error2, sizeof(pred_error2));
        close(fpe2);
        //printf("    | errorPred2: %4.2f", pred_error2);  
        fflush(stdout);

        fpe3 = open(prederrorfifo3, O_RDONLY  );
        float pred_error3  ;
        read(fpe3, &pred_error3, sizeof(pred_error3));
        close(fpe3);
        //printf("    | errorPred3: %4.2f", pred_error3);  
        fflush(stdout);

        //READ kalman estimation error fifo
        fee = open(esterrorfifo, O_RDONLY  );
        float est_error  ;
        read(fee, &est_error, sizeof(est_error));
        close(fee);
        //printf("\nerrorCovPost: %4.2f", est_error);  
        fflush(stdout);  

        fee1 = open(esterrorfifo1, O_RDONLY  );
        float est_error1  ;
        read(fee1, &est_error1, sizeof(est_error1));
        close(fee1);
        //printf("    | errorCovPost1: %4.2f", est_error1);  
        fflush(stdout); 

        fee2 = open(esterrorfifo2, O_RDONLY  );
        float est_error2  ;
        read(fee2, &est_error2, sizeof(est_error2));
        close(fee2);
        //printf("    | errorCovPost2: %4.2f", est_error2);  
        fflush(stdout); 

        fee3 = open(esterrorfifo3, O_RDONLY  );
        float est_error3  ;
        read(fee3, &est_error3, sizeof(est_error3));
        close(fee3);
        //printf("    | errorCovPost3: %4.2f", est_error3);  
        fflush(stdout);  

//        rosfp = open(rospredfifo, O_RDONLY  );          //READ ros kalman prediction fifo
//        float rospred;
//        read(rosfp, &rospred, sizeof(rospred) );
//        close(rosfp);
//        //printf("    | rospred: %4.2f", rospred); 
//        fflush(stdout);
//        
//        rosfp1 = open(rospredfifo1, O_RDONLY  );          //READ ros kalman prediction fifo 2
//        float rospred1;
//        read(rosfp1, &rospred1, sizeof(rospred1) );
//        close(rosfp1);
//        //printf("    | rospred1: %4.2f", rospred1); 
//        fflush(stdout);
//            
//        rosfu = open(rosupdfifo, O_RDONLY  );           //READ ros kalman update fifo
//        float rosupd  ;
//        read(rosfu, &rosupd, sizeof(rosupd) );
//        close(rosfu);
//        //printf("    | rosupd: %4.2f", rosupd);  
//        fflush(stdout);                 // Without this, libc may hold the output in a buffer until the next Mat is read.
//        
//        rosfu1 = open(rosupdfifo1, O_RDONLY  );           //READ ros kalman update fifo
//        float rosupd1  ;
//        read(rosfu1, &rosupd1, sizeof(rosupd1) );
//        close(rosfu1);
//        //printf("    | rosPost1: %4.2f", rosupd1);  
//        fflush(stdout);                 // Without this, libc may hold the output in a buffer until the next Mat is read.
//
//        rosfpe = open(rosprederrorfifo, O_RDONLY  );            //READ ros kalman pred_error fifo
//        float rosprederror;
//        read(rosfpe, &rosprederror, sizeof(rosprederror) );
//        close(rosfpe);
//        //printf("\nrosprederror: %4.2f", rosprederror);       
//        fflush(stdout);
//
//        rosfpe1 = open(rosprederrorfifo1, O_RDONLY  );            //READ ros kalman pred_error fifo
//        float rosprederror1;
//        read(rosfpe1, &rosprederror1, sizeof(rosprederror1) );
//        close(rosfpe1);
//        //printf("    | rosprederror1: %4.2f", rosprederror1);       
//        fflush(stdout);
//
//        rosfpe2 = open(rosprederrorfifo2, O_RDONLY  );            //READ ros kalman pred_error fifo
//        float rosprederror2;
//        read(rosfpe2, &rosprederror2, sizeof(rosprederror2) );
//        close(rosfpe2);
//        //printf("    | rosprederror2: %4.2f", rosprederror2);       
//        fflush(stdout);
//
//        rosfpe3 = open(rosprederrorfifo3, O_RDONLY  );            //READ ros kalman pred_error fifo
//        float rosprederror3;
//        read(rosfpe3, &rosprederror3, sizeof(rosprederror3) );
//        close(rosfpe3);
//        //printf("    | rosprederror3: %4.2f", rosprederror3);       
//        fflush(stdout);
//
//        rosfee = open(rosesterrorfifo, O_RDONLY  );            //READ ros kalman est_error fifo
//        float rosest;
//        read(rosfee, &rosest, sizeof(rosest) );
//        close(rosfee);
//        //printf("\nrosP(k|k): %4.2f", rosest);       
//        fflush(stdout);
//
//        rosfee1 = open(rosesterrorfifo1, O_RDONLY  );            //READ ros kalman est_error fifo
//        float rosest1;
//        read(rosfee1, &rosest1, sizeof(rosest1) );
//        close(rosfee1);
//        //printf("    | rosErrorPost1: %4.2f", rosest1);       
//        fflush(stdout);
//
//        rosfee2 = open(rosesterrorfifo2, O_RDONLY  );            //READ ros kalman est_error fifo
//        float rosest2;
//        read(rosfee2, &rosest2, sizeof(rosest2) );
//        close(rosfee2);
//        //printf("    | rosErrorPost2: %4.2f", rosest2);       
//        fflush(stdout);
//
//        rosfee3 = open(rosesterrorfifo3, O_RDONLY  );            //READ ros kalman est_error fifo
//        float rosest3;
//        read(rosfee3, &rosest3, sizeof(rosest3) );
//        close(rosfee3);
//        //printf("    | rosErrorPost3: %4.2f", rosest3);       
//        fflush(stdout);

        /*Track-to-Track Fusion*/
        
        Vector2f Xbox_update(2,1); 
        Xbox_update  << upd, upd1;

//        Vector2f ROS_update(2,1);
//        ROS_update << rosupd, rosupd1;

        Matrix2f Xbox_Pkk(2,2);
        Xbox_Pkk << est_error, est_error1, est_error2, est_error3;           //covEstimation error

        Matrix2f Xbox_Pkkm1(2, 2); 
        Xbox_Pkkm1 << pred_error, pred_error1, pred_error2, pred_error3;    //covPrediction error

////        Matrix2f ROS_Pkk (2, 2);
////        ROS_Pkk << rosest, rosest1, rosest2, rosest3;   //covEstimation error
////
////        Matrix2f ROS_Pkkm1 (2, 2);
////        ROS_Pkkm1 << rosprederror, rosprederror1, rosprederror2, rosprederror3; //covPrediction error

        /*Track-to-Track Fusion*/
        float fusedpos_ = easytracktotrack(Xbox_update, /*ROS_update,*/ Xbox_Pkk /*ROS_Pkk*/);
        
        boost::asio::io_service io_service;
        sender s(io_service, boost::asio::ip::address::from_string("235.255.0.1"), fusedpos_);
        io_service.run();

        
        now             = std::chrono::high_resolution_clock::now();            
        float deltaT    = chrono::duration_cast<std::chrono::milliseconds>(now - begin).count() / 1000.0;
        cout << "fusion: " << fusedpos_ << "  | deltaT: " << deltaT << "s" << 
                " | fps: " << 1/deltaT << "Hz" << endl;
    }          
    
    /*Delete FIFOs*/
    unlink(predfifo);    
    unlink(predfifo1);

    unlink(updfifo);    
    unlink(updfifo1);

    unlink(prederrorfifo);
    unlink(prederrorfifo1);
    unlink(prederrorfifo2);
    unlink(prederrorfifo3);

    unlink(esterrorfifo);
    unlink(esterrorfifo1);
    unlink(esterrorfifo2);
    unlink(esterrorfifo3);

//    unlink(rospredfifo);
//    unlink(rospredfifo1);
//
//    unlink(rosupdfifo);
//    unlink(rosupdfifo1);
//
//    unlink(rosprederrorfifo);
//    unlink(rosprederrorfifo1);
//    unlink(rosprederrorfifo2);
//    unlink(rosprederrorfifo3);
//
//    unlink(rosesterrorfifo);
//    unlink(rosesterrorfifo1);
//    unlink(rosesterrorfifo2);
//    unlink(rosesterrorfifo3);
//  
    return 0;
}