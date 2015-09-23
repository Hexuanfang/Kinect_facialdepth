###Face Tracker with the Xbox Sensor

####Introduction
Implements the standard **face tracker** and **eye detection** based on **haar cascades**.
The code runs on opencv-openni api and you would need to install the OpenNI library and PrimeSensorModule for OpenNI and
configure OpenCV with WITH_OPENNI flag is ON (using CMake).

####Measurements refinement
A kalman filter is applied on measurements, kalman filter predictions and measurements generated from the face tracker.
The results of the filter are piped to `xbox_listener.cpp` based on the procedure described in [Linux Interprocess communications](http://tldp.org/LDP/lpg/node17.html#SECTION00732000000000000000) where we used a named pipe to move data over the kernel across two processes. In the `xbox_listener` code, the retrieved values are acquired intermittently with the O_NONBLOCK flag after the `xbox_tracker.cpp` has been started. This makes a more efficient transfer of code from one program to the other instead of reading from a text file or streaming over `udp/tcp` connections.

To play with the latency of the tracker code, you could pass the `O_NONBLOCK` flag to the `talker` function and see how you could improve the code.

####Compile and Run

Run the program with the cmake file embedded in the root directory of the file. From the root directory of the project, the following commands will get you up and running:

>cd ../; rm -rf build; mkdir build; cd build; cmake ../; make; cp ../Haar/*.xml `pwd`; ./xbox_tracker -cd 0# Kinect_facialdepth
# Kinect_facialdepth
