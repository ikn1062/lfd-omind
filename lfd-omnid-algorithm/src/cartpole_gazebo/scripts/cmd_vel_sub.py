#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist


def cmd_vel_callback(msg: Twist):
    rospy.loginfo(msg.angular.z)


if __name__ == "__main__":
    rospy.init_node("keyboard_vel_subscriber")
    sub = rospy.Subscriber("/cmd_vel", Twist, callback=cmd_vel_callback)    
    
    rospy.loginfo("Node has been started")
    
    rospy.spin()