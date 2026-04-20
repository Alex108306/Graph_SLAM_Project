# INSTALLATION

This small tutorial is prepared for ubuntu 20.04 and ros noetic, but should work for other versions (replacing all ros-noetic... installs by your ros version)

## Stonefish installation

The stonefish library repository can be found here: [https://github.com/patrykcieslak/stonefish](https://github.com/patrykcieslak/stonefish)
Installation instructions for stonefish can be resumed into:

```bash
 sudo apt install libglm-dev libsdl2-dev libfreetype6-dev #dependencies
 cd (your prefered folder for the repository to download, preferrable not inside your catkin workspace)
 git clone https://github.com/patrykcieslak/stonefish.git # clone repository
 cd stonefish
 mkdir build
 cd build
 cmake ..
 make # or make -j(any number of threads)
 sudo make install
```

## ROS Dependencies

Now its time to install all ROS related packages so you can run the turtlebot simulation. Go inside your catkin workspace src folder and clone the following repositories


**Stonefish ros** is a stonefish simulator with ROS support, the one the turtlebot simulator will depend on. 

```bash
git clone https://github.com/patrykcieslak/stonefish_ros.git
```

**Description packages** basically have the 3D models and urdf files that describe the components necessary for the turtlebot:

```bash
git clone https://bitbucket.com/udg_cirs/kobuki_description.git # Mobile base
git clone https://bitbucket.com/udg_cirs/swiftpro_description.git # Manipulator
git clone https://bitbucket.com/udg_cirs/turtlebot_description.git # Mobile base + manipulator (whole robot)
sudo apt install ros-noetic-realsense2-description # (Realsense camera)
```
**Turtlebot simulation** has the launch files and configurations to launch the turtlebot simulation using stonefish
```bash
git clone https://bitbucket.org/udg_cirs/turtlebot_simulation.git
```

**Extra dependencies:** Before compiling, install ros_control and ros_controllers packages, necessary for controlling the mobile base and the manipulator:

```bash
sudo apt install ros-noetic-ros-control ros-noetic-ros-controllers
```

**Build:** Now you can build your catkin workspace using either catkin_make or catkin build.

Note: Make sure to source your workspace /devel/setup.bash after installation. If it still does not find the packages (bug using catkin_make) just do "rospack list" and everything should be updated.

## RUNNING

You will find three launch files in the turtlebot_simulation package: kobuki_basic.launch, swiftpro_basic.launch and turtlebot_basic.launch.
Launch any of those to run either the mobile base alone, the manipulator alone, or the whole robot, respectively.


## Troubleshooting

We have experienced some students not having xacro installed, though it should be installed with the desktop-full installation of ros. To install xacro just:

```bash
sudo apt install ros-noetic-xacro
```
