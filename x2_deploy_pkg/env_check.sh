#!/usr/bin/env bash
# X2 Sim2Real deployment environment self-check.
# Run on the laptop to confirm the environment can run deploy_x2.
#
# Usage:
#   source /opt/ros/humble/setup.bash
#   source ~/aimdk/install/setup.bash
#   bash env_check.sh

set -u

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

ok() { echo -e "  ${GREEN}[OK]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; FAIL_COUNT=$((FAIL_COUNT+1)); }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; }

FAIL_COUNT=0

echo "=============================================="
echo "  X2 Sim2Real deployment environment self-check"
echo "=============================================="

# ---- 1. Python ----
echo ""
echo "[1] Python"
if command -v python3 >/dev/null 2>&1; then
    PYVER=$(python3 --version 2>&1)
    ok "$PYVER"
else
    fail "python3 not found"
fi

# ---- 2. ROS2 ----
echo ""
echo "[2] ROS2"
if [ -z "${ROS_DISTRO:-}" ]; then
    fail "ROS_DISTRO not set. source /opt/ros/humble/setup.bash first."
else
    ok "ROS_DISTRO=$ROS_DISTRO"
fi

if command -v ros2 >/dev/null 2>&1; then
    ok "ros2 CLI available"
else
    fail "ros2 CLI missing"
fi

# ---- 3. aimdk_msgs ----
echo ""
echo "[3] aimdk_msgs"
python3 -c "from aimdk_msgs.msg import JointCommandArray, JointStateArray, JointCommand" 2>/dev/null
if [ $? -eq 0 ]; then
    ok "aimdk_msgs.msg importable"
else
    fail "aimdk_msgs.msg not importable — source aimdk workspace (e.g. ~/aimdk/install/setup.bash)"
fi

# ---- 4. rclpy + sensor_msgs ----
echo ""
echo "[4] rclpy & sensor_msgs"
python3 -c "import rclpy; from sensor_msgs.msg import Imu" 2>/dev/null
if [ $? -eq 0 ]; then
    ok "rclpy + sensor_msgs importable"
else
    fail "rclpy or sensor_msgs missing"
fi

# ---- 5. onnxruntime ----
echo ""
echo "[5] onnxruntime"
ORT_VER=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null)
if [ -n "$ORT_VER" ]; then
    ok "onnxruntime $ORT_VER"
else
    fail "onnxruntime not installed. Install:  pip3 install --user onnxruntime"
fi

# ---- 6. numpy ----
echo ""
echo "[6] numpy"
NP_VER=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null)
if [ -n "$NP_VER" ]; then
    ok "numpy $NP_VER"
else
    fail "numpy missing. Install: pip3 install --user numpy"
fi

# ---- 7. PyYAML ----
echo ""
echo "[7] PyYAML"
YAML_VER=$(python3 -c "import yaml; print(yaml.__version__)" 2>/dev/null)
if [ -n "$YAML_VER" ]; then
    ok "PyYAML $YAML_VER"
else
    fail "PyYAML not installed. Install:  pip3 install --user PyYAML"
fi

# ---- 8. ONNX model ----
echo ""
echo "[8] models/policy.onnx"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ONNX_PATH="$SCRIPT_DIR/models/policy.onnx"
if [ -L "$ONNX_PATH" ]; then
    REAL=$(readlink -f "$ONNX_PATH")
    if [ -f "$REAL" ]; then
        SIZE=$(stat -c %s "$REAL" 2>/dev/null || stat -f %z "$REAL" 2>/dev/null)
        ok "symlink → $REAL  ($SIZE bytes)"
    else
        fail "symlink broken: $ONNX_PATH → $REAL (target missing)"
    fi
elif [ -f "$ONNX_PATH" ]; then
    SIZE=$(stat -c %s "$ONNX_PATH" 2>/dev/null || stat -f %z "$ONNX_PATH" 2>/dev/null)
    ok "regular file, size=$SIZE bytes"
else
    fail "$ONNX_PATH not found"
fi

if [ -f "$ONNX_PATH" ] || [ -L "$ONNX_PATH" ]; then
    python3 -c "
import onnxruntime as ort
s = ort.InferenceSession('$ONNX_PATH')
inp = s.get_inputs()[0]
out = s.get_outputs()[0]
print(f'  input  {inp.name}: shape={inp.shape} dtype={inp.type}')
print(f'  output {out.name}: shape={out.shape} dtype={out.type}')
" 2>/dev/null
    if [ $? -eq 0 ]; then
        ok "ONNX load test passed"
    else
        fail "ONNX load failed — check model file integrity"
    fi
fi

# ---- 9. deployment scripts ----
echo ""
echo "[9] deployment scripts"
for f in deploy_x2.py \
         x2_config/joint_gains.py x2_config/joint_gains.yaml \
         x2_config/shared/policy_defs.py x2_config/shared/keyboard.py \
         x2_config/deploy/deploy_config.py x2_config/deploy/rc_control_proxy.py; do
    if [ -f "$SCRIPT_DIR/$f" ]; then
        case "$f" in
            *.py)
                python3 -c "import ast; ast.parse(open('$SCRIPT_DIR/$f').read())" 2>/dev/null
                if [ $? -eq 0 ]; then
                    ok "$f syntax OK"
                else
                    fail "$f has syntax error"
                fi
                ;;
            *)
                ok "$f exists"
                ;;
        esac
    else
        fail "$f missing"
    fi
done

# ---- 10. robot topics (optional) ----
echo ""
echo "[10] robot topics (requires the robot powered on + ROS2 network reachable)"
if command -v ros2 >/dev/null 2>&1; then
    echo "  Listing topics under /aima/hal (5s timeout)..."
    AIMA=$(timeout 5 ros2 topic list 2>/dev/null | grep "^/aima/hal" | head -50)
    if [ -n "$AIMA" ]; then
        echo "$AIMA" | while read t; do ok "$t"; done
        EXPECTED=("/aima/hal/joint/leg/state" "/aima/hal/joint/leg/command"
                  "/aima/hal/joint/waist/state" "/aima/hal/joint/waist/command"
                  "/aima/hal/joint/arm/state" "/aima/hal/joint/arm/command"
                  "/aima/hal/joint/head/state" "/aima/hal/joint/head/command"
                  "/aima/hal/imu/torso/state")
        for t in "${EXPECTED[@]}"; do
            echo "$AIMA" | grep -qx "$t" || warn "not found: $t"
        done
    else
        warn "no /aima/hal topics found (robot not powered on, or network not connected, or ROS_DOMAIN_ID mismatch)"
    fi
fi

# ---- 11. RC gamepad integration (optional, only needed with --enable_rc_proxy) ----
# This section uses warn only (never fail): RC is an optional feature enabled only
# with --enable_rc_proxy, so missing dependencies (RC message types, which need the
# full aimdk_msgs) must not fail an ordinary non-RC deployment.
# deploy_to_board.sh aborts the whole deployment on a nonzero env_check exit, so this
# section only warns and does not count toward FAIL.
echo ""
echo "[11] RC gamepad integration (optional: only needed when using --enable_rc_proxy)"
# rc_control_proxy module is importable (shipped together with x2_config/deploy/;
# deploy_x2.py already adds it to sys.path)
# Note: this file's existence and syntax are mandatorily checked in [9]; here we only
# additionally confirm it is importable.
python3 -c "import sys; sys.path.insert(0,'$SCRIPT_DIR/x2_config/deploy'); from rc_control_proxy import RcState, RcControlProxy, action_to_event" 2>/dev/null
if [ $? -eq 0 ]; then
    ok "rc_control_proxy importable"
else
    warn "rc_control_proxy import failed (check $SCRIPT_DIR/x2_config/deploy/rc_control_proxy.py; only affects --enable_rc_proxy)"
fi
# RC-related message/service types are importable
python3 -c "from aimdk_msgs.srv import SetMcAction, GetMcAction; from aimdk_msgs.msg import McLocomotionVelocity, McAction, McActionInfo, McActionStatus" 2>/dev/null
if [ $? -eq 0 ]; then
    ok "aimdk_msgs RC service/message types importable"
else
    warn "aimdk_msgs RC service/message types not importable (only affects --enable_rc_proxy; source a full aimdk_msgs that includes the RC types)"
fi
if command -v ros2 >/dev/null 2>&1; then
    # whether the stock RC's velocity topic is visible (when SOC0 rc is running)
    if timeout 5 ros2 topic list 2>/dev/null | grep -qx "/aima/mc/locomotion/velocity"; then
        ok "found /aima/mc/locomotion/velocity (stock RC is publishing)"
    else
        warn "not found: /aima/mc/locomotion/velocity (stock RC not running, or network/domain mismatch)"
    fi
    # only warn about the real stock MC (node name contains mc): enumerate providers,
    # excluding RC/interaction_slave/ourselves (reuses the rc_control_proxy check)
    MCN=$(timeout 12 python3 -c "
import sys, time; sys.path.insert(0, '$SCRIPT_DIR/x2_config/deploy')
import rclpy; from rc_control_proxy import find_conflicting_mc_providers
rclpy.init(); n = rclpy.create_node('x2_rc_conflict_probe'); time.sleep(2.0)
mc, _ = find_conflicting_mc_providers(n, 'aimdk_5Fmsgs/srv/SetMcAction', self_node_name=n.get_name())
print(','.join(mc))" 2>/dev/null)
    if [ -n "$MCN" ]; then
        warn "stock MC is still providing SetMcAction ($MCN) -- stop the stock MC before enabling --enable_rc_proxy, otherwise the service names collide"
    else
        ok "no stock MC contending for SetMcAction; safe to start the RC proxy"
    fi
fi

# ---- Summary ----
echo ""
echo "=============================================="
if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "  ${GREEN}All checks passed. You can proceed to deployment (deploy_x2.py)${NC}"
else
    echo -e "  ${RED}$FAIL_COUNT check(s) failed; fix them and run again.${NC}"
fi
echo "=============================================="
exit $FAIL_COUNT
