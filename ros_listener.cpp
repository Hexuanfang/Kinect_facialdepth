#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>

#include <string.h>
#include <iostream>
#include <chrono>
#include <Eigen/Dense>

using namespace Eigen;

/*Generate listener for the ros k4w code*/
int main()
{    
/*    int rosfd, rosfm, rosfp, rosfu;
    char * myrosfifo = "/tmp/myrosfifo";

    //Measurement FIFO
    char * rosobsfifo = "/tmp/rosobsfifo";

    //Kalman Prediction FIFO
    char * rospredfifo = "/tmp/rospredfifo";

    //Kalman Update FIFO
    char * rosupdfifo = "/tmp/rosupdfifo";
*/
    /*Xbox Pipes*/
    int fp, fp1, fu, fu1; 
    int fpe, fpe1, fpe2, fpe3;
    int fee, fee1, fee2, fee3;
 //   int fg, fg1;   


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
        printf("    | pred: %4.2f", pred); 
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
        printf("    | update: %4.2f", upd);  
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

        Vector2f Xbox_update(2,1); 
        Xbox_update  << upd, upd1;

        Matrix2f Xbox_Pkk(2,2);
        Xbox_Pkk << est_error, est_error1, est_error2, est_error3;           //covEstimation error
        
        Matrix2f Xbox_Pkkm1(2, 2); 
        Xbox_Pkkm1 << pred_error, pred_error1, pred_error2, pred_error3;    //covPrediction error

        now             = std::chrono::high_resolution_clock::now();            
        float deltaT    = std::chrono::duration_cast<std::chrono::milliseconds>(now - begin).count() / 1000.0;
        printf("\nelapsed time: %4.2f", deltaT);

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

    /*
        while(1)
        {
            //READ actual ros depth fifo
            rosfd = open(myrosfifo, O_RDONLY | O_NONBLOCK);
            float depth; 
            read(rosfd, &depth, sizeof(depth) );            
            close(rosfd);
            printf("\nrosdepth: %4.2f", depth); 

            //READ ros kalman measurement fifo
            rosfm = open(rosobsfifo, O_RDONLY | O_NONBLOCK);
            float obs;
            read(rosfm, &obs, sizeof(obs) );
            close(rosfm);
            printf("    | rosobs: %4.2f", obs);       
            
            //READ ros kalman prediction fifo
            rosfp = open(rospredfifo, O_RDONLY | O_NONBLOCK);
            float pred;
            read(rosfp, &pred, sizeof(pred) );
            close(rosfp);
            printf("    | rospred: %4.2f", pred); 

            //READ ros kalman update fifo
            rosfu = open(rosupdfifo, O_RDONLY | O_NONBLOCK);
            float upd  ;
            read(rosfu, &upd, sizeof(upd) );
            close(rosfu);
            printf("    | rosupdate: %4.2f", upd);  
            // Without this, libc may hold the output in a buffer until the next float is read.
            fflush(stdout);
        }
            */
            /*Delete ROS FIFOs*/
  /*          unlink(myrosfifo);
            
            unlink(rosobsfifo);

            unlink(rospredfifo);

            unlink(rospredfifo);
*/
    return 0;
}