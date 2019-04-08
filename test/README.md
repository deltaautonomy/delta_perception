# Test Scripts

This folder contains some unit tests and sample scripts.

## 1. Test for VehicleGroundTruthArray ROS message

### Setup

The message is defined as follows. It contains the ground truth data for all visible vehilces present in the camera's view.

#### VehicleGroundTruthArray.msg

```
Header header
VehicleGroundTruth[] vehicles
```

#### VehicleGroundTruth.msg

```
string class_name
uint16 left
uint16 top
uint16 right
uint16 bottom
float64 distance
bool difficult
```

#### Descriptions

- std_msgs/string - class name
- std_msgs/uint16 - 2D bbox corners (range: 0 - 65535)
- std_msgs/float64 - radial distance (meters) of vehicle from ego vehicle
- std_msgs/bool - difficult flag, `True`, if vehicle distance > 50m (**subject to change**)

### Sample Usage

Run the following commands to test the message publisher.

```
catkin_make
source devel/setup.bash
rosrun delta_perception vehicle_message_publisher
```

You should be able to echo the `/delta/ground_truth/vehicles` topic. A sample message output is shown below.

```
rostopic echo /delta/ground_truth/vehicles
```

```
---
header: 
  seq: 13
  stamp: 
    secs: 1554700067
    nsecs:  65536975
  frame_id: ''
vehicles: 
  - 
    class_name: "car"
    left: 506
    top: 445
    right: 737
    bottom: 340
    distance: 51.068106185
    difficult: True
  - 
    class_name: "motorbike"
    left: 190
    top: 563
    right: 655
    bottom: 682
    distance: 119.03513482
    difficult: True
  - 
    class_name: "bicycle"
    left: 493
    top: 247
    right: 159
    bottom: 683
    distance: 18.3572980964
    difficult: False
---
```
