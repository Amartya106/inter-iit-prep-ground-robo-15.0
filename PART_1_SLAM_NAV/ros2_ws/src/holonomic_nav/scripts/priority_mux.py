#!/usr/bin/env python3
"""
priority_mux.py -- minimal drop-in replacement for twist_mux.

Multiplexes several geometry_msgs/Twist command streams onto a single
/cmd_vel, always forwarding the highest-priority stream that has published
within its timeout. This is the "one shared drive interface for manual
teleoperation and autonomous navigation" required by Part 1, Phase 1:
a human on the keyboard/gamepad always out-ranks Nav2.

Sources (highest priority first), all remappable:
    ~input_joy  (cmd_vel_joy)  priority 100
    ~input_key  (cmd_vel_key)  priority  90
    ~input_nav  (cmd_vel_nav)  priority  10

If nothing is fresh, a single zero Twist is published so the base stops.

Prefer the real `twist_mux` (ros-jazzy-twist-mux) when available -- it adds
lock topics, per-topic diagnostics and rate limiting. Launch with
mux:=twist_mux to use it instead. This node exists so the package has no
hard apt dependency.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


SOURCES = [
    ("input_joy", "cmd_vel_joy", 100),
    ("input_key", "cmd_vel_key", 90),
    ("input_nav", "cmd_vel_nav", 10),
]


class PriorityMux(Node):
    def __init__(self):
        super().__init__("priority_mux")
        self.declare_parameter("timeout", 0.5)
        self.declare_parameter("rate", 30.0)
        self.timeout = float(self.get_parameter("timeout").value)
        rate = float(self.get_parameter("rate").value)

        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        self._last = {}  # name -> (stamp_sec, Twist)
        self._subs = []
        for name, default_topic, prio in SOURCES:
            topic = default_topic  # remapped by launch if needed
            self._subs.append(
                self.create_subscription(
                    Twist, topic, self._make_cb(name), 10
                )
            )
            self.get_logger().info(f"mux source '{name}' <- {topic} (priority {prio})")

        self._zero_sent = False
        self.create_timer(1.0 / rate, self._tick)

    def _make_cb(self, name):
        def cb(msg):
            self._last[name] = (self.get_clock().now().nanoseconds * 1e-9, msg)
        return cb

    def _tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        for name, _topic, _prio in SOURCES:
            stamp_msg = self._last.get(name)
            if stamp_msg is None:
                continue
            stamp, msg = stamp_msg
            if now - stamp <= self.timeout:
                self.pub.publish(msg)
                self._zero_sent = False
                return
        if not self._zero_sent:
            self.pub.publish(Twist())
            self._zero_sent = True


def main():
    rclpy.init()
    node = PriorityMux()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
