#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Apr 07, 2019
'''

# Python 2/3 compatibility
from __future__ import print_function, absolute_import, division

# Handle paths and OpenCV import
from init_paths import *

# External modules
import matplotlib.pyplot as plt

# ROS modules
import tf
import rospy
import message_filters
from cv_bridge import CvBridge, CvBridgeError

# ROS messages
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image, CameraInfo
from diagnostic_msgs.msg import DiagnosticArray
from delta_msgs.msg import MarkerArrayStamped, CameraTrack, CameraTrackArray
from radar_msgs.msg import RadarTrack, RadarTrackArray
from derived_object_msgs.msg import Object, ObjectArray

# Local python modules
from utils import *
from sort.sort import Sort
from darknet.darknet_video import YOLO
from ipm.ipm import InversePerspectiveMapping
from validator.calculate_map import calculate_map
from cube_marker_publisher import make_cuboid

# Global objects
STOP_FLAG = False
cmap = plt.get_cmap('tab10')
tf_listener = None

# Camera variables
CAMERA_INFO = None
CAMERA_EXTRINSICS = None
CAMERA_PROJECTION_MATRIX = None

# Frames
RADAR_FRAME = '/ego_vehicle/radar'
EGO_VEHICLE_FRAME = 'ego_vehicle'
CAMERA_FRAME = 'ego_vehicle/camera/rgb/front'
VEHICLE_FRAME = 'vehicle/%03d/autopilot'

# Perception models
yolov3 = YOLO()
ipm = InversePerspectiveMapping()
tracker = Sort(max_age=200, min_hits=1, use_dlib=False)

# FPS loggers
FRAME_COUNT = 0
all_fps = FPSLogger('Pipeline')
yolo_fps = FPSLogger('YOLOv3')
sort_fps = FPSLogger('Tracker')


########################### Functions ###########################


def camera_info_callback(camera_info):
    global CAMERA_INFO, CAMERA_PROJECTION_MATRIX
    if CAMERA_INFO is None:
        CAMERA_INFO = camera_info
        CAMERA_PROJECTION_MATRIX = np.matmul(np.asarray(CAMERA_INFO.P).reshape(3, 4), CAMERA_EXTRINSICS)


def validation_setup():
    # Detection results path
    if osp.exists(osp.join(PKG_PATH, 'results/detection-results')):
        shutil.rmtree(osp.join(PKG_PATH,'results/detection-results'), ignore_errors=True)
    os.makedirs(osp.join(PKG_PATH, 'results/detection-results'))

    # Ground thruth results path
    if osp.exists(osp.join(PKG_PATH, 'results/ground-truth')):
        shutil.rmtree(osp.join(PKG_PATH,'results/ground-truth'), ignore_errors=True)
    os.makedirs(osp.join(PKG_PATH, 'results/ground-truth'))

    # Image directory
    if osp.exists(osp.join(PKG_PATH, 'results/images-optional')):
        shutil.rmtree(osp.join(PKG_PATH,'results/images-optional'), ignore_errors=True)
    os.makedirs(osp.join(PKG_PATH, 'results/images-optional'))


def valid_bbox(top_left, bot_right, image_size):
    h, w, c = image_size
    if top_left[0] > 0 and top_left[1] > 0 and bot_right[0] > 0 and bot_right[1] > 0 \
        and top_left[0] < w and top_left[1] < h and bot_right[0] < w and bot_right[1] < h:
        return True
    return False


def validate(image, objects, detections, image_pub, **kwargs):
    global FRAME_COUNT
    FRAME_COUNT += 1

    # Check if camera info is available
    if CAMERA_INFO is None: return

    gt_detects = []
    for obj in objects.objects:
        try:
            # Find the camera to vehicle extrinsics
            (trans, rot) = tf_listener.lookupTransform(CAMERA_FRAME, VEHICLE_FRAME % obj.id, rospy.Time(0))
            # print(trans, np.rad2deg(quaternion_to_rpy(rot)))
            camera_to_vehicle = pose_to_transformation(position=trans, orientation=rot)

            # Project 3D to 2D and filter bbox within image boundaries
            M = np.matrix(CAMERA_INFO.P).reshape(3, 4)
            bbox3D = get_bbox_vertices(camera_to_vehicle, obj.shape.dimensions)
            bbox2D = np.matmul(M, bbox3D.T).T

            # Ignore vehicles behind the camera view
            if bbox2D[0, 2] < 0: continue
            bbox2D = bbox2D / bbox2D[:, -1]
            bbox2D = bbox2D[:, :2].astype('int')

            # Display the 3D bbox vertices on image
            # for point in bbox2D.tolist():
            #     cv2.circle(image, tuple(point), 2, (255, 255, 0), -1)

            # Find the 2D bounding box coordinates
            top_left = (np.min(bbox2D[:, 0]), np.min(bbox2D[:, 1]))
            bot_right = (np.max(bbox2D[:, 0]), np.max(bbox2D[:, 1]))

            # Save ground truth data
            if trans[2] < 100 and valid_bbox(top_left, bot_right, image.shape):
                text = 'car %d %d %d %d %s\n' % (top_left[0], top_left[1], bot_right[0], bot_right[1],
                    'difficult' if (trans[2] < 2 or trans[2] > 60) else '')
                gt_detects.append(text)

                # Draw the rectangle
                # cv2.rectangle(image, top_left, bot_right, (0, 255, 0), 1)
                # cv2.putText(image, 'ID: %d [%.2fm]' % (obj.id, trans[2]), top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2_to_message(image, image_pub)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            # print('Exception', e)
            pass

    # Save image
    # cv2.imwrite(osp.join(PKG_PATH, 'results/images-optional', 'frame_%05d.jpg' % FRAME_COUNT), image)
    
    # Save ground truth
    with open(osp.join(PKG_PATH, 'results/ground-truth/frame_%05d.txt' % FRAME_COUNT), 'w') as f:
        for text in gt_detects: f.write(text)

    # Save detected results
    with open(osp.join(PKG_PATH, 'results/detection-results/frame_%05d.txt' % FRAME_COUNT), 'w') as f:
        for detection in detections:
            label = detection[0]
            confidence = detection[1]
            xmin, ymin, xmax, ymax = detection[2]
            text = '%s %.5f %d %d %d %d\n' % (label, confidence, xmin, ymin, xmax, ymax)
            f.write(text)


def visualize(img, tracked_targets, detections, radar_targets, publishers, **kwargs):
    # Draw visualizations
    # img = YOLO.cvDrawBoxes(detections, img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display tracked targets
    for tracked_target, detection in zip(tracked_targets, detections):
        label, score, bbox = detection
        x1, y1, x2, y2, tracker_id = tracked_target.astype('int')
        color = tuple(map(int, (np.asarray(cmap(tracker_id % 10))[:-1] * 255).astype('uint8')))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, '%s [%d%%] [ID: %d]' % (label.decode('utf-8').title(), score * 100, tracker_id),
            (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # Project the radar points on the image
    for track_id, uv, pos, vel in radar_targets:
        uv = np.asarray(uv).flatten().tolist()
        cv2.circle(img, tuple(uv), 10, (0, 0, 255), 3)
        cv2.putText(img, '[P: %.2fm]' % (pos[0]),
            tuple(uv), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.putText(img, '[V: %.2fKm/h]' % (vel * 3.6), 
            (uv[0], uv[1] + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        # Publish radar track marker for debugging.
        radar_marker = make_cuboid(position=pos, scale=[0.5] * 3,
            frame_id=RADAR_FRAME, marker_id=track_id, duration=0.5, color=[1, 0, 0])
        publishers['radar_marker_pub'].publish(radar_marker)

    cv2_to_message(img, publishers['image_pub'])


def get_radar_targets(radar_msg):
    # Project the radar points on image
    uv_points = []
    for track in radar_msg.tracks:
        if CAMERA_PROJECTION_MATRIX is not None:
            pos_msg = position_to_numpy(track.track_shape.points[0])
            pos = np.asarray([pos_msg[0], pos_msg[1], pos_msg[2]])
            pos = np.matrix(np.append(pos, 1)).T
            uv = np.matmul(CAMERA_PROJECTION_MATRIX, pos)
            uv = uv / uv[-1]
            uv = uv[:2].astype('int').tolist()
            uv_points.append((track.track_id, uv, pos[:3], track.linear_velocity.x))
    return uv_points


def publish_camera_tracks(publishers, tracked_targets, detections, timestamp):
    camera_track_array = CameraTrackArray()
    camera_track_array.header.stamp = timestamp
    camera_track_array.header.frame_id = EGO_VEHICLE_FRAME
    
    # Populate camera track message.
    for target, detection in zip(tracked_targets, detections):
        label, score, _ = detection
        u1, v1, u2, v2, track_id = target
        # todo(heethesh): Verify u/v convention here
        x, y = ipm.transform_points_to_m([(u1 + u2) / 2, (v1 + v2) / 2])  # Bottom mid-point of bbox
        camera_track = CameraTrack()
        camera_track.x = y
        camera_track.y = -x
        camera_track.track_id = track_id
        camera_track.label = label
        camera_track.confidence = score
        camera_track_array.tracks.append(camera_track)

        # Publish camera track marker for debugging.
        camera_marker = make_cuboid(position=[y, -x, 0], scale=[0.5] * 3,
            frame_id=EGO_VEHICLE_FRAME, marker_id=track_id, duration=0.5, color=[0, 0, 1])
        publishers['camera_marker_pub'].publish(camera_marker)

    # Publish the camera tarck data.
    publishers['tracker_pub'].publish(camera_track_array)


def publish_diagnostics(publishers):
    msg = DiagnosticArray()
    msg.header.stamp = rospy.Time.now()
    msg.status.append(make_diagnostics_status('object_detection', 'perception', str(yolo_fps.fps)))
    msg.status.append(make_diagnostics_status('bounding_box_tracking', 'perception', str(sort_fps.fps)))
    publishers['diag_pub'].publish(msg)


def perception_pipeline(img, radar_msg, publishers, vis=True, **kwargs):
    # Log pipeline FPS
    all_fps.lap()

    # Preprocess
    # img = increase_brightness(img)

    # Object detection
    yolo_fps.lap()
    detections, frame_resized = yolov3.run(img)
    yolo_fps.tick()

    # Object tracking
    sort_fps.lap()
    dets = np.asarray([[bbox[0], bbox[1], bbox[2], bbox[3], score]
                       for label, score, bbox in detections])
    tracked_targets = tracker.update(dets, img)
    sort_fps.tick()

    # Publish camera tracks
    publish_camera_tracks(publishers, tracked_targets,
        detections, radar_msg.header.stamp)

    # RADAR tracking
    radar_targets = get_radar_targets(radar_msg)

    # Display FPS logger status
    all_fps.tick()
    # sys.stdout.write('\r%s | %s | %s ' % (all_fps.get_log(),
    #     yolo_fps.get_log(), sort_fps.get_log()))
    # sys.stdout.flush()

    # Publish diagnostics status
    publish_diagnostics(publishers)

    # Visualize and publish image message
    if vis: visualize(img, tracked_targets, detections, radar_targets, publishers)

    return detections


def perception_callback(image_msg, radar_msg, objects, publishers, **kwargs):
    # Node stop has been requested
    if STOP_FLAG: return

    # Read image message
    img = message_to_cv2(image_msg)
    if img is None:
        print('Error')
        sys.exit(1)

    # Run the perception pipeline
    detections = perception_pipeline(img.copy(), radar_msg, publishers)

    # Run the validation pipeline
    validate(img.copy(), objects, detections, publishers['image_pub'])


def shutdown_hook():
    global STOP_FLAG
    STOP_FLAG = True
    time.sleep(3)
    print('\n\033[95m' + '*' * 30 + ' Delta Perception Shutdown ' + '*' * 30 + '\033[00m\n')
    print('\n\033[95m' + '*' * 30 + ' Calculating YOLOv3 mAP ' + '*' * 30 + '\033[00m\n')
    print('YOLOv3 Mean Average Precision @ 0.5 Overlap: %.3f%%\n' % (calculate_map(0.5) * 100))
    # print('YOLOv3 Mean Average Precision @ 0.7 Overlap: %.3f%%\n' % (calculate_map(0.7) * 100))


def run(**kwargs):
    global tf_listener, CAMERA_EXTRINSICS

    # Start node
    rospy.init_node('main', anonymous=True)
    rospy.loginfo('Current PID: [%d]' % os.getpid())
    tf_listener = tf.TransformListener()

    # Setup validation
    validation_setup()
    
    # Setup models
    yolov3.setup()

    # Find the camera to vehicle extrinsics
    tf_listener.waitForTransform(CAMERA_FRAME, RADAR_FRAME, rospy.Time(), rospy.Duration(100.0))
    (trans, rot) = tf_listener.lookupTransform(CAMERA_FRAME, RADAR_FRAME, rospy.Time(0))
    CAMERA_EXTRINSICS = pose_to_transformation(position=trans, orientation=rot)

    # Handle params and topics
    camera_info = rospy.get_param('~camera_info', '/carla/ego_vehicle/camera/rgb/front/camera_info')
    image_color = rospy.get_param('~image_color', '/carla/ego_vehicle/camera/rgb/front/image_color')
    object_array = rospy.get_param('~object_array', '/carla/objects')
    radar = rospy.get_param('~radar', '/carla/ego_vehicle/radar/tracks')
    output_image = rospy.get_param('~output_image', '/delta/perception/object_detection/image')
    camera_track = rospy.get_param('~camera_track', '/delta/perception/ipm/camera_track')
    camera_track_marker = rospy.get_param('~camera_track_marker', '/delta/perception/camera_track_marker')
    radar_track_marker = rospy.get_param('~radar_track_marker', '/delta/perception/radar_track_marker')
    diagnostics = rospy.get_param('~diagnostics', '/delta/perception/object_detection/diagnostics')

    # Display params and topics
    rospy.loginfo('CameraInfo topic: %s' % camera_info)
    rospy.loginfo('Image topic: %s' % image_color)
    rospy.loginfo('RADAR topic: %s' % radar)
    rospy.loginfo('CameraTrackArray topic: %s' % camera_track)
    rospy.loginfo('Diagnostics topic: %s' % diagnostics)

    # Publish output topic
    publishers = {}
    publishers['image_pub'] = rospy.Publisher(output_image, Image, queue_size=5)
    publishers['tracker_pub'] = rospy.Publisher(camera_track, CameraTrackArray, queue_size=5)
    publishers['camera_marker_pub'] = rospy.Publisher(camera_track_marker, Marker, queue_size=5)
    publishers['radar_marker_pub'] = rospy.Publisher(radar_track_marker, Marker, queue_size=5)
    publishers['diag_pub'] = rospy.Publisher(diagnostics, DiagnosticArray, queue_size=5)

    # Subscribe to topics
    info_sub = rospy.Subscriber(camera_info, CameraInfo, camera_info_callback)
    image_sub = message_filters.Subscriber(image_color, Image)
    radar_sub = message_filters.Subscriber(radar, RadarTrackArray)
    object_sub = message_filters.Subscriber(object_array, ObjectArray)

    # Synchronize the topics by time
    ats = message_filters.ApproximateTimeSynchronizer(
        [image_sub, radar_sub, object_sub], queue_size=1, slop=0.5)
    ats.registerCallback(perception_callback, publishers, **kwargs)

    # Shutdown hook
    rospy.on_shutdown(shutdown_hook)

    # Keep python from exiting until this node is stopped
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')


if __name__ == '__main__':
    # Start perception node
    run()
