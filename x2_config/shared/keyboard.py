"""Shared keyboard-teleop plumbing for X2 sim2sim + deploy.

Extracts the fiddly, previously-duplicated terminal handling (raw/cbreak mode,
non-blocking select loop, restore-on-exit) and the velocity-stepping clamp.
Each script keeps its own per-key dispatch (`handle_key`) since the bindings
differ (sim has push keys + local state; deploy has FSM gating + ROS node),
and each keeps its own HELP_TEXT.
"""

import select
import sys
import termios
import tty


# Per-script key legends (printed on '?'); kept here so keyboard content lives in one file.
SIM_HELP_TEXT = """
[Sim2Sim keys]  (same as deploy_x2.py)
  FSM:
    Enter/Space    DAMPING -> JOINT_DEFAULT (standard forward)
    Backspace      any RL/transition state -> DAMPING (emergency stop)
    Esc            DAMPING/ESTOP -> exit

  RL_ACTIVE velocity control:
    W/S            forward/backward +/-step (first W jumps 0 -> initial_cmd_vx)
    A/D            left/right +/-vy_step
    Q/E            turn left/right +/-vyaw_step
    Space          zero all velocities (stand in place)

  External pushes (100ms impulse):
    I              push FORWARD (100N)
    K              push BACKWARD (100N)
    J              push LEFT (100N)
    L              push RIGHT (100N)
    U              push RANDOM direction (150N)
    1-5            push scale (1=100N, 5=500N)

  ?              print this help
"""

DEPLOY_HELP_TEXT = """
\033[1mX2 Deploy keys\033[0m
  FSM:
    Enter/Space ①  DAMPING -> JOINT_DEFAULT (half-squat)
    Enter/Space ②  JOINT_DEFAULT (after interp done) -> STANDUP -> RL_ACTIVE
    Backspace      any RL/transition state -> DAMPING (emergency brake)
    Esc            DAMPING/ESTOP/WAITING -> exit program

  RL_ACTIVE velocity control:
    W/S          forward/backward ±step (first W jumps 0 -> initial_cmd_vx)
    A/D          left/right ±vy_step
    Q/E          turn left/right ±vyaw_step
    Space        zero all velocities (stand in place)

  ?             print this help
  Ctrl+C        any state -> DAMPING + exit
"""


def step_velocity(value, delta, lo, hi):
    """Clamped velocity step: max(lo, min(value + delta, hi))."""
    return max(lo, min(value + delta, hi))


class KeyListener:
    """Non-blocking single-key reader. Each keypress dispatches to `handle_key`
    immediately (no Enter needed) and the terminal is restored on exit.

    Args:
        handle_key: callable(str) invoked per keypress.
        cbreak: True -> tty.setcbreak (keeps OPOST/ONLCR and ISIG so Ctrl+C still
            exits and log \\n aligns); False -> tty.setraw + stdout \\n->\\r\\n
            rewrite (needed when raw mode disables output post-processing).
        poll: select() timeout in seconds.
        should_continue: callable()->bool loop guard (default: run forever).
    """

    def __init__(self, handle_key, *, cbreak, poll=0.05, should_continue=None):
        self.handle_key = handle_key
        self.cbreak = cbreak
        self.poll = poll
        self.should_continue = should_continue or (lambda: True)

    def run(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        old_write = sys.stdout.write

        def _raw_write(s):
            old_write(s.replace("\n", "\r\n"))
            sys.stdout.flush()

        try:
            if self.cbreak:
                tty.setcbreak(fd)
            else:
                tty.setraw(fd)
                sys.stdout.write = _raw_write
            while self.should_continue():
                ready, _, _ = select.select([sys.stdin], [], [], self.poll)
                if ready:
                    ch = sys.stdin.read(1)
                    if ch:
                        self.handle_key(ch)
        finally:
            sys.stdout.write = old_write
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
