#!/usr/bin/env bash
# Usage (run from any dir; the script cd's to repo root itself):
#   x2_deploy_pkg/deploy_to_board.sh <robot-user> <robot-ip> [board-dir, default x2_deploy]
# Optional: export X2_ONNX_MD5=<expected-md5> to enforce a strict model check (guards against deploying the wrong model).
set -euo pipefail

# Self-locate to repo root (this script is under x2_deploy_pkg/; FILES paths are relative to repo root).
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."

USER="${1:?Usage: deploy_to_board.sh <robot-user> <robot-ip> [dest]}"
IP="${2:?need <robot-ip>}"
DEST="${3:-x2_deploy}"                 # deploy dir on board, relative to home
FILES=(
  x2_deploy_pkg/deploy_x2.py
  x2_deploy_pkg/env_check.sh
)
# x2_config/ ships as a whole dir (scp -r below); deploy_x2.py puts its subdirs on sys.path.
# These config files must exist locally before syncing:
CONFIG_PKG_FILES=(
  x2_config/joint_gains.py
  x2_config/joint_gains.yaml
  x2_config/shared/policy_defs.py
  x2_config/shared/keyboard.py
  x2_config/deploy/deploy_config.py
  x2_config/deploy/rc_control_proxy.py
)
MODEL="x2_deploy_pkg/models/policy.onnx"        # local source path (may be nested)
REMOTE_MODEL="models/$(basename "$MODEL")"       # board-relative path: always models/<basename>

# 0. Local files present + model md5
for f in "${FILES[@]}" "${CONFIG_PKG_FILES[@]}" "$MODEL"; do
  [ -f "$f" ] || { echo "x missing file at repo root: $f (are you in the mc-rl repo root?)"; exit 1; }
done
LOCAL_MD5=$(md5sum "$MODEL" | awk '{print $1}')
echo "local $MODEL md5 = $LOCAL_MD5"
if [ -n "${X2_ONNX_MD5:-}" ] && [ "$LOCAL_MD5" != "$X2_ONNX_MD5" ]; then
  echo "x local model md5 differs from expected ($X2_ONNX_MD5), aborting"; exit 1
fi

# 1. Sync to board (keep models/ path; deploy_x2.py loads models/policy.onnx by default)
echo "-> syncing to $USER@$IP:~/$DEST"
ssh "$USER@$IP" "mkdir -p ~/$DEST/models"
scp -O "${FILES[@]}" "$USER@$IP:~/$DEST/"    # -O: classic SCP protocol, widest compatibility (incl. servers with sftp-server disabled)
scp -O -r x2_config  "$USER@$IP:~/$DEST/"    # ship x2_config/ dir (deploy_x2.py puts its subdirs on sys.path)
scp -O "$MODEL"      "$USER@$IP:~/$DEST/models/"

# 2. Verify md5 match on board (guards against transfer corruption / stale copy)
REMOTE_MD5=$(ssh "$USER@$IP" "md5sum ~/$DEST/$REMOTE_MODEL | awk '{print \$1}'")
echo "board $REMOTE_MODEL md5 = $REMOTE_MD5"
[ "$LOCAL_MD5" = "$REMOTE_MD5" ] || { echo "x board model md5 differs from local, aborting"; exit 1; }
echo "v md5 match"

# 3. Source + env_check on board
echo "running env_check.sh on board ..."
if ssh "$USER@$IP" "source /opt/ros/humble/setup.bash && source ~/aimdk/install/setup.bash && cd ~/$DEST && bash env_check.sh"; then
  echo "v ready. Next (manual): ssh to board -> cd ~/$DEST -> run deploy_x2.py to enter FSM."
  echo "  note: on first run / after stop, waist not publishing can stall sensor-ready -> start-app mc && sleep 2 && stop-app mc first."
else
  echo "x env_check has failures, fix before deploying."; exit 1
fi
