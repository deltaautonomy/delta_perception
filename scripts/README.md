# Perception Pipeline

This package has two ROS nodes for object detection and tracking (`main.py`), and lane detection (`lane_detection.py`).

## Object Detection and Tracking Node

### Topics

Following are the input topics to this node.
- `/carla/ego_vehicle/camera/rgb/front/camera_info` of type `sensor_msgs.CameraInfo`.
- `/carla/ego_vehicle/camera/rgb/front/image_color` of type `sensor_msgs.Image`.
- `/carla/ego_vehicle/radar/tracks` of type `radar_msgs.RadarTrackArray`.

Following are the output topics from this node.
- `/delta/perception/object_detection_tracking/image` of type `sensor_msgs.Image`. This image displays the result of object detection and tracking on the input image.
- `/delta/perception/ipm/camera_track` of type `delta_perception.CameraTrackArray`. This message contains the IPM projected centroids of object bounding boxes and calibrated to meters.

The following output topics are used only for visualization purposes.
- `/delta/perception/camera_track_marker` of type `visualization_msgs.Marker`. This message is used to visualize IPM projected camera detections as cube markers on Rviz.
- `/delta/perception/radar_track_marker` of type `visualization_msgs.Marker`. This message is used to visualize RADAR detections as cube markers on Rviz.
- `/delta/perception/occupancy_grid` of type `nav_msgs.OccupancyGrid`. (Deprecated) this message contains the RADAR detections (no covariance) on the occupancy grid.

### Usage

Run the following command to execute this node.
```
rosrun delta_perception main.py
```

## Lane Detection Node

### Topics

Following are the input topics to this node.
- `/carla/ego_vehicle/camera/rgb/front/image_color` of type `sensor_msgs.Image`.

Following are the output topics from this node.
- `/delta/perception/lane_detection/image` of type `sensor_msgs.Image`. This message displays the output of lane detection on the input image.
- `/delta/perception/lane_detection/markings` of type `delta_perception.LaneMarkingArray`. This message contains the slope and intercept data (in meters) of the detected lanes.
- `/delta/perception/lane_detection/occupancy_grid` of type `nav_msgs.OccupancyGrid`. (Deprecated) this message contains the lane detection data on occupancy grid with a resolution of 20cm.

### Usage

Run the following command to execute this node.
```
rosrun delta_perception lane_detection.py

```
