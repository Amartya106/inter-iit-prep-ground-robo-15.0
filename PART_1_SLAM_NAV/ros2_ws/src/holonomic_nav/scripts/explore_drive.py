#!/usr/bin/env python3
"""
explore_drive.py -- reactive autonomous exploration for the SLAM mapping run.

Purpose: a *reproducible, headless* trajectory for building the occupancy
grid, so the mapping run never depends on a human driving. It is not a
navigation algorithm -- Nav2 does the real point-to-point navigation once a
map exists. Commands go out on /cmd_vel_key (teleop priority) so they pass
through the same twist_mux / priority_mux as manual driving.

Two things shape the design:

  1. Platform. The holonomic_bot's "mecanum" wheels fake omni motion with an
     anisotropic-friction trick and in Gazebo the base is NOT holonomic:
     linear.y (strafe) produces almost no motion and angular.z yaws much
     slower than commanded, with a forward drift. So the controller only
     uses linear.x + angular.z, and treats heading as something it can bias
     but not set. NOTE: an earlier version of this docstring additionally
     claimed a specific ~10x gain and an inverted sign on angular.z -- that
     was never actually implemented anywhere in this file (self.bias is
     used with plain right-hand-rule sign throughout) and was never
     verified either way, so it's removed here rather than left as an
     unverified claim. The stuck-recovery logic below (see BACKUP state)
     is deliberately direction/sign-agnostic for exactly this reason: it
     does not need to know which way is actually open, it alternates and
     backs up until it clears.

  2. Sensor vs world. The LiDAR reaches 12 m; the warehouse is ~30 x 34 m.
     From the open interior every return is `inf`, so SLAM only ever sees a
     wall when the robot is within ~12 m of one. The trajectory is therefore
     a *wall follower*: hug the nearest structure at ~6-8 m stand-off (inside
     LiDAR range, structure always in view) and circle the layout. Only when
     something gets too close does it bounce with a turn pulse.

Runs for `duration` seconds (0 = until Ctrl-C), then stops.

    ros2 run holonomic_nav explore_drive.py --ros-args -p duration:=200.0
    ros2 run holonomic_nav explore_drive.py --ros-args -p standoff:=7.0 -p forward_speed:=0.6
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


class ExploreDrive(Node):
    def __init__(self):
        super().__init__("explore_drive")
        self.declare_parameter("forward_speed", 0.6)      # m/s along the wall
        self.declare_parameter("turn_speed", 1.4)         # rad/s during a bounce turn
        self.declare_parameter("turn_creep", 0.12)        # m/s kept on during a bounce
        self.declare_parameter("turn_min_time", 3.5)      # s, minimum bounce length
        self.declare_parameter("standoff", 7.0)           # m, target wall stand-off
        self.declare_parameter("bounce_distance", 2.4)    # m, anything closer -> bounce
        self.declare_parameter("resume_distance", 3.2)    # m, front clear -> follow again
        self.declare_parameter("reacquire_distance", 11.0)  # m, farther -> steer back to wall
        self.declare_parameter("side_bias", 1.0)          # +1 wall on the left, -1 right
        self.declare_parameter("turn_max_time", 9.0)      # s, hard cap on one TURN attempt
        self.declare_parameter("backup_speed", 0.25)      # m/s reverse, during BACKUP
        self.declare_parameter("backup_time", 1.5)        # s, duration of a BACKUP pulse
        self.declare_parameter("stuck_reset_time", 6.0)   # s, FOLLOW run this long clears
                                                           # the stuck-escalation counter
        self.declare_parameter("duration", 0.0)           # s, 0 = until Ctrl-C
        self.declare_parameter("cmd_topic", "/cmd_vel_key")
        # accepted-but-ignored (older callers / launch files):
        self.declare_parameter("speed", 0.0)
        self.declare_parameter("clear_distance", 0.0)
        self.declare_parameter("waypoints_file", "")
        self.declare_parameter("wp_tol", 0.0)
        self.declare_parameter("laps", 0)

        self.v = float(self.get_parameter("forward_speed").value)
        self.w = float(self.get_parameter("turn_speed").value)
        self.creep = float(self.get_parameter("turn_creep").value)
        self.turn_min = float(self.get_parameter("turn_min_time").value)
        self.standoff = float(self.get_parameter("standoff").value)
        self.bounce = float(self.get_parameter("bounce_distance").value)
        self.resume = float(self.get_parameter("resume_distance").value)
        self.reacq = float(self.get_parameter("reacquire_distance").value)
        self.bias = 1.0 if float(self.get_parameter("side_bias").value) >= 0 else -1.0
        self.duration = float(self.get_parameter("duration").value)
        self.turn_max = float(self.get_parameter("turn_max_time").value)
        self.backup_speed = float(self.get_parameter("backup_speed").value)
        self.backup_time = float(self.get_parameter("backup_time").value)
        self.stuck_reset_time = float(self.get_parameter("stuck_reset_time").value)
        cmd_topic = self.get_parameter("cmd_topic").value

        self._scan = None
        self._front = math.inf
        self.state = "FOLLOW"
        self._turn_t0 = None
        self._backup_t0 = None
        self._follow_t0 = None      # when the current FOLLOW streak began
        self._stuck_count = 0       # consecutive bounce/turn/backup escalations
        self.pub = self.create_publisher(Twist, cmd_topic, 10)
        self.create_subscription(LaserScan, "/scan", self._scan_cb, 10)
        self._t0 = self.get_clock().now()
        self.create_timer(0.1, self._tick)
        self.get_logger().info(
            f"exploring [wall-follow]: v={self.v} standoff={self.standoff}m "
            f"bias={'L' if self.bias > 0 else 'R'} "
            f"duration={'inf' if self.duration == 0 else self.duration}s -> {cmd_topic}"
        )

    def _scan_cb(self, m: LaserScan):
        self._scan = m
        self._front = self._sector_min(0.0, math.radians(25.0))

    def _sector_min(self, centre: float, half: float) -> float:
        m = self._scan
        if m is None:
            return math.inf
        n = len(m.ranges)
        lo = int((centre - half - m.angle_min) / m.angle_increment)
        hi = int((centre + half - m.angle_min) / m.angle_increment)
        vals = [m.ranges[i % n] for i in range(lo, hi + 1)
                if m.range_min < m.ranges[i % n] < m.range_max and m.ranges[i % n] == m.ranges[i % n]]
        return min(vals) if vals else math.inf

    def _nearest(self):
        """(range, bearing) of the closest return over the whole circle."""
        m = self._scan
        if m is None:
            return math.inf, 0.0
        n = len(m.ranges)
        step = max(1, n // 180)
        best_r, best_a = math.inf, 0.0
        for i in range(0, n, step):
            r = m.ranges[i]
            if m.range_min < r < best_r and r < m.range_max and r == r:
                best_r, best_a = r, m.angle_min + i * m.angle_increment
        return best_r, best_a

    def _tick(self):
        now = self.get_clock().now()
        if self.duration > 0 and (now - self._t0).nanoseconds * 1e-9 >= self.duration:
            self.pub.publish(Twist())
            self.get_logger().info("exploration duration reached; stopping")
            rclpy.shutdown()
            return

        cmd = Twist()
        if self._scan is None:
            self.pub.publish(cmd)
            return

        # bounce whenever the forward sector is blocked. TURN's exit used to
        # be "minimum hold time AND clear front" with NO maximum -- if the
        # 0.12 m/s forward creep (below) closes distance to the obstacle
        # faster than a fixed turn direction sweeps the 50deg forward cone
        # clear (a real trap in corners / along long walls, not just an
        # unlucky local minimum), `held >= turn_min` becomes true but
        # `_front > resume` never does, and the identical command repeats
        # forever -- confirmed: no odometry/progress-tracking exists
        # anywhere in this file, so the old code had no way to notice.
        # Fix: a hard turn_max_time ceiling escalates into BACKUP (an
        # actual reverse pulse -- nothing here ever commanded reverse
        # before) instead of repeating; each BACKUP physically increases
        # distance from the obstacle, so it cannot loop forever the way
        # TURN alone could. The escalation also alternates turn direction
        # (`_stuck_count` parity) so it doesn't matter whether `side_bias`
        # happens to be the wrong way for this particular obstacle -- no
        # need to know, or guess, which way is actually open.
        if self.state == "FOLLOW" and self._front < self.bounce:
            # a FOLLOW run long enough to count as real progress clears the
            # escalation counter, so a fresh bounce starts at attempt 1
            # (normal side_bias direction, no backup) rather than staying
            # escalated from a stuck episode resolved a while ago
            if (self._follow_t0 is not None
                    and (now - self._follow_t0).nanoseconds * 1e-9 >= self.stuck_reset_time):
                self._stuck_count = 0
            self._stuck_count += 1
            self.state = "TURN"
            self._turn_t0 = now

        if self.state == "TURN":
            held = (now - self._turn_t0).nanoseconds * 1e-9
            if held >= self.turn_min and self._front > self.resume:
                self.state = "FOLLOW"
                self._follow_t0 = now
            elif held >= self.turn_max:
                # this attempt failed to clear in time -- back up for real
                # instead of repeating the same turn command indefinitely
                self.get_logger().warn(
                    f"TURN timed out after {held:.1f}s (attempt {self._stuck_count}), "
                    f"backing up and reversing direction")
                self.state = "BACKUP"
                self._backup_t0 = now
            else:
                # odd attempts use the configured side_bias unchanged (no
                # behavior change from before in the common, non-stuck
                # case); even attempts (i.e. every retry after a timeout)
                # flip it
                eff_bias = self.bias if self._stuck_count % 2 == 1 else -self.bias
                cmd.angular.z = eff_bias * self.w
                cmd.linear.x = self.creep
                self.pub.publish(cmd)
                return

        if self.state == "BACKUP":
            held = (now - self._backup_t0).nanoseconds * 1e-9
            if held >= self.backup_time:
                # try TURN again, with the direction flipped relative to the
                # attempt that just timed out (increment here, not just on
                # the FOLLOW->TURN entry above, or every post-backup retry
                # would repeat the same failed direction forever)
                self._stuck_count += 1
                self.state = "TURN"
                self._turn_t0 = now
                # publish this tick's TURN command directly rather than
                # falling through into the FOLLOW steering code below --
                # nothing after this checks `if self.state == "TURN"` again
                # this tick, unlike the FOLLOW->TURN transition above
                eff_bias = self.bias if self._stuck_count % 2 == 1 else -self.bias
                cmd.angular.z = eff_bias * self.w
                cmd.linear.x = self.creep
                self.pub.publish(cmd)
                return
            else:
                cmd.linear.x = -self.backup_speed
                self.pub.publish(cmd)
                return

        # FOLLOW: drive forward; steer to hold the nearest structure abeam at
        # `standoff`. Wall is kept on the `bias` side (left for bias>0), i.e.
        # near a bearing of +pi/2*bias in the body frame.
        near_r, near_a = self._nearest()
        cmd.linear.x = self.v
        if near_r > self.reacq:
            # lost the wall -- curl toward the bias side to find one again
            cmd.angular.z = self.bias * 0.4
        else:
            want = self.bias * math.pi / 2.0
            abeam_err = math.atan2(math.sin(near_a - want), math.cos(near_a - want))
            dist_err = near_r - self.standoff        # +ve: too far from wall
            # steer: close the abeam angle, and bias the heading in/out to
            # correct stand-off (toward the wall if too far, away if too close)
            steer = 1.2 * abeam_err - self.bias * 0.15 * max(-3.0, min(3.0, dist_err))
            cmd.angular.z = max(-self.w, min(self.w, steer))
        self.pub.publish(cmd)


def main():
    rclpy.init()
    node = ExploreDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
