"""Sim-to-sim validation: run an IL4OP policy in MuJoCo.

Example for a Go2W policy trained to track a base-roll command:
    python tools/sim2sim/play.py --robot go2w --policy PATH_TO_POLICY.pt --obs_dim 61 \
        --roll 0.2 --command 0.5 0 0 --duration 15 --viewer

For a flat-roll policy, --roll is a target trunk roll in radians. It is inserted into the
policy observation in the same position as the trained command in Isaac Lab; it does not
directly tilt the MuJoCo model. Use a policy trained with the roll-command observation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from robot import ROBOTS, Sim2Sim, Teleop, teleoperate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default="go1", choices=sorted(ROBOTS))
    parser.add_argument("--policy", type=Path,
                        help="exported policy.pt; without it only the Unitree controller runs")
    parser.add_argument("--obs_dim", type=int, default=45, help="input size of the policy")
    parser.add_argument("--nominal", default="current", choices=("current", "legacy"),
                        help="the pose the policy was trained around; 'legacy' is the Isaac Lab"
                             " default, used by the Go1 policies trained before STAND_POSE")
    parser.add_argument("--height", type=float, default=0.33,
                        help="base height command of the Go2W flat-z policies (0.25 - 0.40)")
    parser.add_argument("--roll", type=float, default=None,
                        help="target base-roll command in radians for a Go2W flat-roll policy")
    parser.add_argument("--command", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                        metavar=("VX", "VY", "WZ"), help="base velocity command")
    parser.add_argument("--duration", type=float, default=10.0, help="seconds to simulate")
    parser.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer")
    parser.add_argument("--teleop", action="store_true",
                        help="drive the robot from the keyboard")
    parser.add_argument("--control", default="sticky", choices=Teleop.MODES,
                        help="how the keyboard drives the robot (see the key map)")
    parser.add_argument("--keys", default="terminal", choices=("terminal", "viewer"),
                        help="where the keys are read from; the viewer binds most letters itself")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="scale the speeds the keyboard may ask a policy for, up to the"
                             " range it was trained on (1.0 is near what the gait does)")
    args = parser.parse_args()

    if args.policy is None and not args.teleop:
        parser.error("--policy is needed for a scripted run; --teleop works without one")

    if args.roll is not None:
        if args.robot != "go2w":
            parser.error("--roll is available only for the Go2W flat-roll policy")
        if args.obs_dim != 61:
            parser.error("--roll requires --obs_dim 61, matching Go2WFlatRollObservationsCfg")
        if not -0.4 <= args.roll <= 0.4:
            parser.error("--roll must be between -0.4 and 0.4 radians, the range used during training")
        if args.teleop:
            parser.error("--roll is a scripted policy command and cannot be combined with --teleop")

    sim = Sim2Sim(args.robot, args.policy, args.obs_dim, args.nominal, roll_command=args.roll)
    sim.height_command = args.height
    sim.reset()

    if args.teleop:
        teleoperate(sim, args.height, args.control, args.keys, args.speed)
        return

    command = np.array(args.command)

    dt = sim.model.opt.timestep * sim.cfg["decimation"]
    steps = int(args.duration / dt)
    heights, vel_error, torques, yaw_rate, rolls = [], [], [], [], []
    roll_errors = []
    start_xy = sim.data.qpos[:2].copy()
    fell_at = None

    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sim.model, sim.data)

    for i in range(steps):
        sim.step(command)
        lin_vel_b, ang_vel_b = sim.base_velocity()
        roll, _ = sim.attitude()
        rolls.append(roll)
        if args.roll is not None:
            roll_error = (roll - args.roll + np.pi) % (2.0 * np.pi) - np.pi
            roll_errors.append(abs(roll_error))
        heights.append(sim.data.qpos[2])
        vel_error.append(abs(lin_vel_b[0] - command[0]))
        yaw_rate.append(ang_vel_b[2])
        torques.append(np.abs(sim.data.ctrl[sim.act_idx]).mean())
        if fell_at is None and sim.data.qpos[2] < sim.cfg["fall_height"]:
            fell_at = i * dt
        if viewer is not None:
            viewer.sync()

    if viewer is not None:
        viewer.close()

    print(f"\npolicy   : {args.policy}")
    print(f"robot    : {args.robot}   command: vx={command[0]} vy={command[1]} wz={command[2]}")
    if args.roll is not None:
        print(f"roll command: {args.roll:+.3f} rad ({math.degrees(args.roll):+.1f} deg)")
    print(f"simulated: {steps * dt:.1f} s at {1 / dt:.0f} Hz")
    print("-" * 56)
    if fell_at is None:
        print(f"  stayed up for the whole run")
    else:
        print(f"  FELL at {fell_at:.2f} s (base below {sim.cfg['fall_height']} m)")
    print(f"  base height   mean {np.mean(heights):.3f} m   min {np.min(heights):.3f} m")
    print(f"  base roll     mean {np.mean(rolls):+.3f} rad   max abs {np.max(np.abs(rolls)):.3f} rad")
    if roll_errors:
        print(f"  roll error    mean {np.mean(roll_errors):.3f} rad   max {np.max(roll_errors):.3f} rad")
    print(f"  |vx - cmd|    mean {np.mean(vel_error):.3f} m/s")
    print(f"  |torque|      mean {np.mean(torques):.2f} Nm   max {np.max(torques):.2f} Nm")
    travelled = sim.data.qpos[:2] - start_xy
    print(f"  travelled     {travelled[0]:+.2f} m forward, {travelled[1]:+.2f} m sideways")
    print(f"  yaw rate      mean {np.mean(yaw_rate):+.3f} rad/s (command {command[2]:+.2f})")


if __name__ == "__main__":
    main()
