#!/usr/bin/env python3
from __future__ import print_function

import rospy
from std_msgs.msg import Float64

import sys
import os
from select import select
import threading
import termios
import tty

moveBindings = {'a': -1, 'd': 1, 'A': -1, 'D': 1}
magBindings = {'w': 2.0, 's': 0.5, 'W': 2.0, 'S': 0.5}


class PublishThread(threading.Thread):
    def __init__(self, rate):
        """
        Creates a publisher to the cart_pole_controller/command and publishes key strokes from terminal to forces,
        upadtes magnitude when magnitude bindings are changed

        :param rate: Rate at which new messages are published
        """
        super(PublishThread, self).__init__()

        self.publisher = rospy.Publisher("/cart_pole_controller/command", Float64, queue_size=1)
        self.force = 0
        self.f = 0.0
        self.mag = 0.0

        self.condition = threading.Condition()
        self.done = False

        # Set timeout to None if rate is 0 (causes new_message to wait forever for new data to publish)
        if rate != 0.0:
            self.timeout = 1.0 / rate
        else:
            self.timeout = None

        self.start()

    def update(self, f, mag):
        """
        Notifies publish thread that we have a new message from key stroke

        :param f: Force from moveBindings (int)
        :param mag: Magnitude from magBindings (int)
        :return: None
        """
        self.condition.acquire()
        self.f, self.mag = f, mag
        # Notify publish thread that we have a new message.
        self.condition.notify()
        self.condition.release()

    def stop(self):
        """
        Stops all threads

        :return: None
        """
        self.done = True
        self.update(0, 0)
        self.join()

    def run(self):
        """
        Updates and publishes current force based of key stroke

        - Current force is sticky and is sent as the key is pressed down

        :return: None
        """
        while not self.done:
            self.condition.acquire()
            self.condition.wait(self.timeout)

            # add brackets here
            if self.f == 0.0 or self.f < 0 and self.force > 0 or self.f > 0 and self.force < 0:
                self.force = 0
            else:
                self.force += self.f * self.mag

            if self.force > 240:
                self.force = 240.0
            elif self.force < -240:
                self.force = -240.0

            # print(force)
            self.publisher.publish(self.force)
            self.condition.release()

        # Publish stop message when thread exits.
        self.force = 0
        self.publisher.publish(self.f)


def getKey(settings, timeout):
    """
    Gets current key typed into terminal

    :param settings: Terminal Settings
    :param timeout: Key Timeout (rospy param)
    :return: Key (str)
    """
    tty.setraw(sys.stdin.fileno())
    # sys.stdin.read() returns a string on Linux
    rlist, _, _ = select([sys.stdin], [], [], timeout)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def vels(mag):
    """
    Prints veloctity message with current magnitude of force

    :param mag: Magnitude (int)
    :return: Message (str)
    """
    return "currently:\tmagnitude %s\n " % (mag)


def saveTerminalSettings():
    """
    Saves Terminal Message from linux terminal

    :return: Linux Terminal Settings
    """
    return termios.tcgetattr(sys.stdin)


def restoreTerminalSettings(old_settings):
    """
    Restore Terminal Settings

    :param old_settings: Linux Terminal Setting
    :return: None
    """
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    settings = saveTerminalSettings()

    rospy.init_node('key_control_cart')

    mag = rospy.get_param("~mag", 1.0)
    repeat = rospy.get_param("~repeat_rate", 5.0)
    key_timeout = rospy.get_param("~key_timeout", 0.5)

    pub_thread = PublishThread(repeat)

    f = 0

    try:
        pub_thread.update(f, mag)
        rospy.loginfo(vels(mag))
        while True:
            # Gets key from terminal
            key = getKey(settings, key_timeout)

            # Updates force/magnitude if key maps to force/magnitude
            if key in moveBindings.keys():
                f = moveBindings[key]
            elif key in magBindings.keys():
                mag = mag * magBindings[key]
                rospy.loginfo(vels(mag))
            else:
                if key == '' and f == 0:
                    continue
                f = 0
                if key == '\x03':
                    break

            # Publishes to rospy
            pub_thread.update(f, mag)

    except Exception as e:
        rospy.logerr(e)

    finally:
        pub_thread.stop()
        restoreTerminalSettings(settings)