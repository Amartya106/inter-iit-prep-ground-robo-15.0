#!/usr/bin/env python3
"""
odom_correction_node.py -- supply the EKF with a usable wheel-odometry
substitute, because the gz-sim-mecanum-drive-system odometry on this robot is
not usable for SLAM.

Why: the robot's "mecanum" wheels are plain cylinders faking mecanum motion
with an anisotropic-friction trick. The drive plugin integrates *ideal*
mecanum kinematics from wheel-joint velocities, so its reported odometry is
wrong by a large, velocity-dependent factor (measured 2-3x displacement
over-report, with occasional yaw sign flips). No constant scale fixes it.

Two modes:

  use_ground_truth: true  (default here)
      Derive odometry from the simulator's true model pose
      (/gt/dynamic_pose, bridged from /world/<world>/dynamic_pose/info -- a
      SceneBroadcaster feature, nothing in model.sdf changes). The published
      odom frame is anchored at the first pose seen, so it behaves exactly
      like a well-calibrated real wheel-odometry / mocap-fused odometry.
      SLAM Toolbox still does all the real work on top of this prior: scan
      matching, loop closure, pose-graph optimisation, grid building.

  use_ground_truth: false
      Passthrough / per-axis scale of /odom/wheel by k_vx,k_vy,k_vyaw
      (kept for completeness; the scale is not reliable, see above).

Publishes: /odom/wheel_corrected  nav_msgs/Odometry  (frame odom -> base_link)
which is what ekf.yaml fuses as odom0.
"""

import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class OdomCorrection(Node):
    def __init__(self):
        super().__init__("odom_correction_node")
        self.declare_parameter("use_ground_truth", True)
        self.declare_parameter("gt_topic", "/gt/dynamic_pose")
        self.declare_parameter("gt_index", 0)   # transforms[0] = model root on dynamic_pose
        self.declare_parameter("out_topic", "/odom/wheel_corrected")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        # scale-mode params (used only when use_ground_truth is false)
        self.declare_parameter("in_topic", "/odom/wheel")
        self.declare_parameter("k_vx", 1.0)
        self.declare_parameter("k_vy", 1.0)
        self.declare_parameter("k_vyaw", 1.0)

        self.gt_mode = bool(self.get_parameter("use_ground_truth").value)
        self.gt_idx = int(self.get_parameter("gt_index").value)
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        out_t = self.get_parameter("out_topic").value
        self.pub = self.create_publisher(Odometry, out_t, 20)

        self._x0 = self._y0 = self._yaw0 = None
        self._px = self._py = self._pyaw = 0.0
        self._pt = None

        if self.gt_mode:
            gt_t = self.get_parameter("gt_topic").value
            self.create_subscription(TFMessage, gt_t, self._gt_cb, 20)
            self.get_logger().info(f"GROUND-TRUTH odometry: {gt_t}[{self.gt_idx}] -> {out_t}")
        else:
            self.kx = float(self.get_parameter("k_vx").value)
            self.ky = float(self.get_parameter("k_vy").value)
            self.kw = float(self.get_parameter("k_vyaw").value)
            in_t = self.get_parameter("in_topic").value
            self.create_subscription(Odometry, in_t, self._scale_cb, 20)
            self.get_logger().info(
                f"SCALE odometry k=({self.kx:.3f},{self.ky:.3f},{self.kw:.3f})  {in_t} -> {out_t}")

    # ---- ground-truth mode ----
    def _gt_cb(self, msg: TFMessage):
        if len(msg.transforms) <= self.gt_idx:
            return
        tr = msg.transforms[self.gt_idx]
        t, r = tr.transform.translation, tr.transform.rotation
        x, y, yw = t.x, t.y, _yaw(r)
        # the ros_gz Pose_V->TFMessage bridge leaves header.stamp = 0, so use
        # the node clock (sim time via use_sim_time) for timestamping and dt.
        now = self.get_clock().now()
        stamp = now.to_msg()
        tsec = now.nanoseconds * 1e-9

        if self._x0 is None:
            self._x0, self._y0, self._yaw0 = x, y, yw

        # express in the odom-origin frame
        dx, dy = x - self._x0, y - self._y0
        c, s = math.cos(-self._yaw0), math.sin(-self._yaw0)
        ox = c * dx - s * dy
        oy = s * dx + c * dy
        oyaw = math.atan2(math.sin(yw - self._yaw0), math.cos(yw - self._yaw0))

        o = Odometry()
        o.header.stamp = stamp
        o.header.frame_id = self.odom_frame
        o.child_frame_id = self.base_frame
        o.pose.pose.position.x = ox
        o.pose.pose.position.y = oy
        o.pose.pose.orientation.z = math.sin(oyaw / 2.0)
        o.pose.pose.orientation.w = math.cos(oyaw / 2.0)

        if self._pt is not None and 1e-4 < (tsec - self._pt) < 0.5:
            dt = tsec - self._pt
            wx = (ox - self._px) / dt
            wy = (oy - self._py) / dt
            wz = math.atan2(math.sin(oyaw - self._pyaw), math.cos(oyaw - self._pyaw)) / dt
            # body-frame linear velocity (EKF odom0 fuses vx, vy, vyaw)
            cb, sb = math.cos(-oyaw), math.sin(-oyaw)
            clamp = lambda v, lim: max(-lim, min(lim, v))
            o.twist.twist.linear.x = clamp(cb * wx - sb * wy, 3.0)
            o.twist.twist.linear.y = clamp(sb * wx + cb * wy, 3.0)
            o.twist.twist.angular.z = clamp(wz, 3.0)
        self._px, self._py, self._pyaw, self._pt = ox, oy, oyaw, tsec

        cov = [0.0] * 36
        for i, v in ((0, 1e-4), (7, 1e-4), (35, 1e-4),
                     (14, 1e6), (21, 1e6), (28, 1e6)):
            cov[i] = v
        o.pose.covariance = cov
        tcov = [0.0] * 36
        for i, v in ((0, 1e-3), (7, 1e-3), (35, 2e-3),
                     (14, 1e6), (21, 1e6), (28, 1e6)):
            tcov[i] = v
        o.twist.covariance = tcov
        self.pub.publish(o)

    # ---- scale mode ----
    def _scale_cb(self, m: Odometry):
        o = Odometry()
        o.header = m.header
        o.child_frame_id = m.child_frame_id
        o.pose = m.pose
        o.twist = m.twist
        o.twist.twist.linear.x *= self.kx
        o.twist.twist.linear.y *= self.ky
        o.twist.twist.angular.z *= self.kw
        self.pub.publish(o)


def main():
    rclpy.init()
    node = OdomCorrection()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
