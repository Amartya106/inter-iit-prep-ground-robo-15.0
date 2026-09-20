#!/usr/bin/python3
"""
Splits a standard body-frame /cmd_vel Twist into the three Float64 velocity
commands consumed by the robot's virtual holonomic joints in Gazebo Harmonic
(joint_x, joint_y, joint_yaw -- see description/holonomic_drive_gz.xacro).

    /cmd_vel.linear.x  -> /joint_x_cmd    (forward / body-frame X)
    /cmd_vel.linear.y  -> /joint_y_cmd    (strafe  / body-frame Y)
    /cmd_vel.angular.z -> /joint_yaw_cmd  (yaw rate)

These three ROS topics are bridged to gz Double topics by
config/bridge_harmonic.yaml, which the gz-sim-joint-controller-system
plugins on each joint subscribe to.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64


class CmdVelDecompose(Node):
    def __init__(self):
        super().__init__('cmd_vel_decompose')
        self.sub = self.create_subscription(Twist, 'cmd_vel', self.cb, 10)
        self.pub_x = self.create_publisher(Float64, 'joint_x_cmd', 10)
        self.pub_y = self.create_publisher(Float64, 'joint_y_cmd', 10)
        self.pub_yaw = self.create_publisher(Float64, 'joint_yaw_cmd', 10)

    def cb(self, msg: Twist):
        self.pub_x.publish(Float64(data=msg.linear.x))
        self.pub_y.publish(Float64(data=msg.linear.y))
        self.pub_yaw.publish(Float64(data=msg.angular.z))


def main():
    rclpy.init()
    node = CmdVelDecompose()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
