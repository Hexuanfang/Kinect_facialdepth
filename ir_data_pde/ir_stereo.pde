/* --------------------------------------------------------------------------
 * SimpleOpenNI IR Test
 * --------------------------------------------------------------------------
 * Processing Wrapper for the OpenNI/Kinect library
 * http://code.google.com/p/simple-openni
 * --------------------------------------------------------------------------
 * prog:  Max Rheiner / Interaction Design / zhdk / http://iad.zhdk.ch/
 * date:  02/16/2011 (m/d/y)
 * ----------------------------------------------------------------------------
 */

import SimpleOpenNI.*;


SimpleOpenNI  context;
//static int i  =0;
void setup()
{
  context = new SimpleOpenNI(this);
 
  // enable depthMap generation 

  
  // enable ir generation
  if(context.enableIR() == false)
  {
     println("Can't open the depthMap, maybe the camera is not connected!"); 
     exit();
     return;
  }
  
  background(200,0,0);
  size( context.irWidth() , context.irHeight()); 
}

void draw()
{
  // update the cam
  context.update();
    
  // draw depthImageMap
  //image(context.depthImage(),0,0);
  
  // draw irImageMap
  image(context.irImage(),0,0);
}

void keyPressed() {
  if (key == ENTER) {
  
   saveFrame("ir-stereo.png");
   //i++;
  }}
