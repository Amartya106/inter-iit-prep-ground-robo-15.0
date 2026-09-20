#!/usr/bin/env python3
"""
pose_error_logger.py -- DIAGNOSTIC: score the estimated trajectory against
Gazebo ground truth during a mapping run.

Subscribes:
  /gt/dynamic_pose      tf2_msgs/TFMessage  (transforms[gt_index] = model root)
                        -- ground truth, bridged from /world/warehouse/dynamic_pose/info
  /odom/wheel           nav_msgs/Odometry   (raw mecanum-drive odometry; optional)
  /odometry/filtered    nav_msgs/Odometry   (EKF output)
and looks up TF  map -> base_link          (SLAM-corrected pose), if available.

Writes one CSV row per tick, flushed immediately (so a hard kill during
teardown never loses the log):
  t, gt_x, gt_y, gt_yaw, odo_x, odo_y, odo_yaw, ekf_x, ekf_y, ekf_yaw,
  slam_x, slam_y, slam_yaw, path_len

On SIGINT it also prints a summary: final position error and max |heading
error| for raw-odom / EKF / SLAM vs ground truth, plus the raw-odom error
split into along-track (scale) vs cross-track (heading).

    ros2 run holonomic_nav pose_error_logger.py --ros-args -p out:=/tmp/pose_err.csv
"""

import csv
import math

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
import tf2_ros

NAN = float("nan")


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class PoseErrorLogger(Node):
    def __init__(self):
        super().__init__("pose_error_logger")
        self.declare_parameter("out", "/tmp/pose_err.csv")
        self.declare_parameter("rate", 10.0)
        self.declare_parameter("gt_index", 0)
        self.out = self.get_parameter("out").value
        rate = float(self.get_parameter("rate").value)
        self.gt_idx = int(self.get_parameter("gt_index").value)

        self.gt = None
        self.gt0 = None
        self.odo = None
        self.ekf = None
        self._rows = []
        self._path_len = 0.0
        self._prev_gt_xy = None

        self._fh = open(self.out, "w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(["t", "gt_x", "gt_y", "gt_yaw", "odo_x", "odo_y", "odo_yaw",
                          "ekf_x", "ekf_y", "ekf_yaw", "slam_x", "slam_y", "slam_yaw",
                          "path_len"])
        self._fh.flush()

        self.tfbuf = tf2_ros.Buffer()
        self.tflist = tf2_ros.TransformListener(self.tfbuf, self)
        self.create_subscription(TFMessage, "/gt/dynamic_pose", self._gt_cb, 20)
        self.create_subscription(Odometry, "/odom/wheel", self._odo_cb, 20)
        self.create_subscription(Odometry, "/odometry/filtered", self._ekf_cb, 20)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(f"logging pose error -> {self.out}")

    def _gt_cb(self, msg: TFMessage):
        if len(msg.transforms) <= self.gt_idx:
            return
        tr = msg.transforms[self.gt_idx]
        t, r = tr.transform.translation, tr.transform.rotation
        self.gt = (t.x, t.y, _yaw(r))
        if self.gt0 is None:
            self.gt0 = self.gt

    def _odo_cb(self, m: Odometry):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.odo = (p.x, p.y, _yaw(q))

    def _ekf_cb(self, m: Odometry):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.ekf = (p.x, p.y, _yaw(q))

    def _slam_pose(self):
        try:
            tr = self.tfbuf.lookup_transform("map", "base_link", rclpy.time.Time())
            t, r = tr.transform.translation, tr.transform.rotation
            return (t.x, t.y, _yaw(r))
        except Exception:
            return (NAN, NAN, NAN)

    def _tick(self):
        if self.gt is None or self.ekf is None:
            return
        gx = self.gt[0] - self.gt0[0]
        gy = self.gt[1] - self.gt0[1]
        gyaw = _wrap(self.gt[2] - self.gt0[2])
        if self._prev_gt_xy is not None:
            self._path_len += math.hypot(gx - self._prev_gt_xy[0], gy - self._prev_gt_xy[1])
        self._prev_gt_xy = (gx, gy)
        odo = self.odo if self.odo is not None else (NAN, NAN, NAN)
        sx, sy, syaw = self._slam_pose()
        t = self.get_clock().now().nanoseconds * 1e-9
        row = [t, gx, gy, gyaw, *odo, *self.ekf, sx, sy, syaw, self._path_len]
        self._rows.append(row)
        self._w.writerow([f"{v:.5f}" for v in row])
        self._fh.flush()

    def finish(self):
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
        if not self._rows:
            self.get_logger().warn("no rows logged")
            return
        a = np.array(self._rows, dtype=float)
        gt, odo, ekf, slam = a[:, 1:4], a[:, 4:7], a[:, 7:10], a[:, 10:13]
        plen = a[-1, 13]

        def report(name, est):
            if not np.isfinite(est).any():
                return
            fin = math.hypot(est[-1, 0] - gt[-1, 0], est[-1, 1] - gt[-1, 1])
            dyaw = np.arctan2(np.sin(est[:, 2] - gt[:, 2]), np.cos(est[:, 2] - gt[:, 2]))
            maxyaw = np.nanmax(np.abs(dyaw))
            meanpos = np.nanmean(np.hypot(est[:, 0] - gt[:, 0], est[:, 1] - gt[:, 1]))
            print(f"  {name:9s} final pos err {fin:5.2f} m  mean {meanpos:4.2f} m  "
                  f"({100*fin/max(plen,1e-6):4.1f}% of {plen:.1f} m path)  "
                  f"max |yaw err| {math.degrees(maxyaw):5.1f} deg", flush=True)

        d_gt = gt[-1, :2] - gt[0, :2]
        L = float(np.linalg.norm(d_gt))
        if L > 1e-6 and np.isfinite(odo[-1]).all():
            u = d_gt / L
            e = odo[-1, :2] - gt[-1, :2]
            along = float(np.dot(e, u))
            cross = float(e[0] * -u[1] + e[1] * u[0])
            print(f"  raw-odom error split: along-track {along:+.2f} m "
                  f"({100*along/max(L,1e-6):+.1f}% -> odom SCALE), "
                  f"cross-track {cross:+.2f} m (-> HEADING)", flush=True)
        print(f"path length {plen:.1f} m; rows {len(self._rows)}; csv {self.out}", flush=True)
        report("raw-odom", odo)
        report("EKF", ekf)
        report("SLAM", slam)


def main():
    rclpy.init()
    node = PoseErrorLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
