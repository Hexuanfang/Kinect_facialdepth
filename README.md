###Face Tracker with the Xbox Sensor

####Project Update
Work Progressed: total hours ( 10x6 hours)
-Understanding the fundamental concepts involved in this project.
-Understanding the preliminary code.
-Translation of the code to run only on CPU support to match the host computer specifications.
-Correction of the code for precise Depth Measurement.
-Included the formulation for World XYZ computation baseof default Intrinsic and Extrinsic calibration values.

In Progress:
-Estimation of precise Intrinsic and Extrinsic calibration values.
-Alternative inter-process communication through LCM.

To be done:
-Roll Pitch Yaw Estimation.
-Final fixes in the code.

####Compile and Run
To run the program. cd into Kinect_facialdepth; cmake . ; sudo make; ./xbox_listener in one terminal and ./xbox_tracker in another.
# Kinect_facialdepth
