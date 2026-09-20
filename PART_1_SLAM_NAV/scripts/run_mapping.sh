#!/bin/bash
# Headless end-to-end mapping run:
#   warehouse_sim (sim + drive interface + EKF + ground-truth truth-bridge +
#                  odom_correction_node)  +  online_slam (slam_toolbox)
#   +  pose_error_logger  +  autonomous LiDAR wall-follow drive
#   ->  map_saver  +  pose-graph serialize  +  morphological-open de-speckle
#
# Usage:   scripts/run_mapping.sh [OUT_DIR] [DRIVE_SECONDS]
#   OUT_DIR         where to write warehouse.{yaml,pgm,posegraph.*}  (default: ../maps)
#   DRIVE_SECONDS   autonomous drive length in *sim* time            (default: 450)
#
# Note: on a machine that simulates this world below real time, wall-clock is
# proportionally longer. A longer drive thickens walls (see report §4.4).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
WS="$HERE/../ros2_ws"
OUT=${1:-"$HERE/../maps"}
DRIVE_T=${2:-450}
LOG=${LOG:-/tmp/run_mapping_logs}
mkdir -p "$OUT" "$LOG"
# Canonicalize to an absolute path NOW -- the script cd's into $WS below,
# and a relative $OUT (e.g. the documented "scripts/run_mapping.sh maps 900"
# invoked from PART_1_SLAM_NAV/) would silently resolve against the WRONG
# directory for every later use (rm -f, map_saver, serialize_map), each
# failing or no-op'ing against a nonexistent ros2_ws/maps/ instead of the
# intended directory. Confirmed: this is exactly what happened on a real
# run -- map_saver errored "Unable to open file", serialize_map returned
# result=255, and the pre-existing map was untouched only because the `rm
# -f` step hit the same wrong (nonexistent) path and no-op'd.
OUT=$(cd "$OUT" && pwd)
# ROS 2's own setup.bash references AMENT_TRACE_SETUP_FILES (and similar)
# without defaulting them, which is fine under normal bash but fatal under
# `set -u` above ("unbound variable") -- disable -u for just the sourcing,
# same standard workaround used anywhere ROS setup scripts meet set -u.
set +u
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
set -u
cd "$WS"

kill_pat() { for p in $(pgrep -f "$1" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done; }
PRE='gz sim -r|gz_tools_vendor/bin/gz sim|ros2 launch holonomic|async_slam_toolbox|ekf_node|ros_gz_bridge/parameter_bridge|odom_correction_node|priority_mux|explore_drive.py|pose_error_logger|robot_state_publisher --ros-args --params-file /tmp'
IFS='|'; for p in $PRE; do kill_pat "$p"; done; unset IFS
sleep 3

# No `setsid`: with `setsid X &`, $! is the transient setsid pid (double fork),
# useless for killing. Plain `& PID=$!` gives the real launch pid; SIGINT lets
# it shut its own nodes down, then a pgrep sweep mops up strays.
ros2 launch holonomic_nav warehouse_sim.launch.py headless:=true use_rviz:=false > "$LOG/sim.log" 2>&1 &
SIM=$!
ros2 launch holonomic_nav online_slam.launch.py > "$LOG/slam.log" 2>&1 &
SLAM=$!
teardown() {
  kill -INT "$SLAM" "$SIM" 2>/dev/null; sleep 4
  IFS='|'; for p in $PRE'|ruby.*gz_tools_vendor'; do kill_pat "$p"; done; unset IFS
  sleep 2; kill_pat "gz sim -r"
}
trap teardown EXIT

for i in $(seq 1 60); do timeout 3 ros2 topic echo --once /scan >/dev/null 2>&1 && break; sleep 2; done
echo ">>> scan up"
for i in $(seq 1 40); do
  timeout 4 ros2 topic echo --once /map 2>/dev/null | grep -q "resolution:" && { echo ">>> map active"; break; }
  sleep 1
done

ros2 run holonomic_nav pose_error_logger.py --ros-args -p out:="$LOG/pose_err.csv" > "$LOG/poselog.log" 2>&1 &

echo ">>> autonomous wall-follow drive (${DRIVE_T}s sim-time)"
timeout $((DRIVE_T * 18 + 180)) ros2 run holonomic_nav explore_drive.py --ros-args \
  -p forward_speed:=0.6 -p turn_speed:=1.4 -p turn_min_time:=3.5 \
  -p standoff:=7.0 -p bounce_distance:=2.4 -p resume_distance:=3.4 \
  -p reacquire_distance:=11.0 -p side_bias:=1.0 -p duration:=${DRIVE_T}.0 \
  > "$LOG/explore.log" 2>&1
echo ">>> drive finished"
kill -INT "$(pgrep -f 'lib/holonomic_nav/pose_error_logger.py')" 2>/dev/null; sleep 6

echo ">>> loop closures: $(grep -c -iE 'loop clos|closing loop' "$LOG/slam.log" 2>/dev/null || echo 0)"
echo ">>> pose-error summary:"; grep -aE "final pos err|error split|path length" "$LOG/poselog.log" 2>/dev/null

rm -f "$OUT"/warehouse.pgm "$OUT"/warehouse.yaml
ros2 run nav2_map_server map_saver_cli -f "$OUT/warehouse" \
  --ros-args -p save_map_timeout:=30.0 -p free_thresh:=0.25 -p occupied_thresh:=0.65 > "$LOG/save.log" 2>&1
echo ">>> map_saver exit $?"
ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph \
  "{filename: '$OUT/warehouse_posegraph'}" > "$LOG/serialize.log" 2>&1

python3 - "$OUT/warehouse.pgm" "$OUT/warehouse_map.png" <<'PY'
import sys, numpy as np
from PIL import Image
pgm, png = sys.argv[1], sys.argv[2]
a = np.array(Image.open(pgm)); H, W = a.shape
occ = (a < 65).astype(np.uint8)
def sh(x, dr, dc):
    o = np.zeros_like(x)
    o[max(dr,0):H+min(dr,0), max(dc,0):W+min(dc,0)] = x[max(-dr,0):H+min(-dr,0), max(-dc,0):W+min(-dc,0)]
    return o
er = occ.copy()
for dr in (-1,0,1):
    for dc in (-1,0,1): er = er & sh(occ, dr, dc)          # erode
di = er.copy()
for dr in (-1,0,1):
    for dc in (-1,0,1): di = di | sh(er, dr, dc)           # dilate  -> morphological open
removed = int((occ & ~di).sum())
b = a.copy(); b[(occ == 1) & (di == 0)] = 254
Image.fromarray(b).save(pgm)
occ2 = (b < 65).sum(); free = (b > 250).sum(); tot = b.size
print(f">>> map {W}x{H}  occ {100*occ2/tot:.1f}%  free {100*free/tot:.1f}%  "
      f"known {100*(occ2+free)/tot:.1f}%  (de-speckled {removed} px)")
Image.fromarray(b).convert('L').resize(
    (min(W,760), min(H, int(760*H/max(W,1))))).save(png)
PY
echo ">>> DONE  ->  $OUT"
