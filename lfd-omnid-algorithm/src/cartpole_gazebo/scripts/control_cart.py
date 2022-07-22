#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist

force = 0.0


def cmd_effort(msg: Twist):
    global force
    
    if msg.angular.z != 0.0:
        force += msg.angular.z * 2.0
    else:
        force = 0.0
        
    rospy.loginfo(f"Force: {force}")
    pub.publish(force)


if __name__ == "__main__":
    rospy.init_node("cart_effort_controller")
    
    pub = rospy.Publisher("/cart_pole_controller/command", Float64, queue_size=1)
    sub = rospy.Subscriber("/cmd_vel", Twist, callback=cmd_effort)
    
    rospy.loginfo("Effort Controller Node has been started")
    rospy.spin()