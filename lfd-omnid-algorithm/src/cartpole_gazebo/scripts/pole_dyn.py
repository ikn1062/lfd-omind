#!/usr/bin/env python3
import rospy
from math import pi
from sensor_msgs.msg import JointState
import csv


# Global vars
csv_writer = None


def pole_state(msg: JointState):    
    theta, theta_dot = msg.position[0], msg.velocity[0]
    pos_x, vel_x = msg.position[1], msg.velocity[1]
    
    while theta < -2 * pi:
        theta += 2 * pi
    while theta > 2 * pi:
        theta -= 2 * pi
    
    output = f"(t: {theta}, td: {theta_dot}), x: {pos_x}, xd: {vel_x}"

    rospy.loginfo(output)
    
    if csv_writer != None:
        csv_writer.writerow([pos_x, vel_x, theta, theta_dot])


def main():
    rospy.init_node("pole_state_sub")
    sub = rospy.Subscriber("/cart_pole/joint_states", JointState, callback=pole_state)    
    rospy.loginfo("Pole State Node has been started")

    f = open('/home/ishaan/catkin_ws/src/cartpole_gazebo/dynamics/test2.csv', 'w')
    global csv_writer
    # csv_writer = csv.writer(f)
        
    rospy.spin()
    
    f.close()
    csv_writer = None
    

if __name__ == "__main__":
    main()
