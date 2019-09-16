#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Apr 08, 2019
'''

import random
import rospy
from delta_perception.msg import VehicleGroundTruth, VehicleGroundTruthArray

random_vehicles = ['car', 'truck', 'bus', 'motorbike', 'bicycle']

def talker():
    # Setup node
    pub = rospy.Publisher('/delta/ground_truth/vehicles', VehicleGroundTruthArray, queue_size=10)
    rospy.init_node('vehicle_ground_truth_publisher_test', anonymous=True)
    
    # Ignore this if you use a camera/image callback to publish the data
    r = rospy.Rate(0.25)

    # Randomly publish some data
    while not rospy.is_shutdown():
        # Create the message array
        msg = VehicleGroundTruthArray()

        # Create few random vehicles
        for i in range(random.randrange(5)):
            # Populate single vehicle with random ground truth data
            vehicle = VehicleGroundTruth()

            # std_msgs/string - class name
            vehicle.class_name = random.choice(random_vehicles)
            # std_msgs/uint16 - 2D bbox corners (range: 0 - 65535)
            vehicle.left = random.randint(0, 1000)
            vehicle.top = random.randint(0, 1000)
            vehicle.right = random.randint(0, 1000)
            vehicle.bottom = random.randint(0, 1000)
            # std_msgs/float64 - radial distance (m) of vehicle from ego vehicle
            vehicle.distance = random.random() * 150
            # std_msgs/bool - difficult flag, True, if vehicle distance > 50m
            vehicle.difficult = vehicle.distance > 50

            # Add the vehicle to the message array
            msg.vehicles.append(vehicle)

        # Header stamp and publish the message
        print('Sending')
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)

        # Ignore this if you use a camera/image callback to publish the data
        r.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
