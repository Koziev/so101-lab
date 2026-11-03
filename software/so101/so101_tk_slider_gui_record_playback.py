#!/usr/bin/env python3
"""Simple Tkinter slider controller for a calibrated LeRobot SO-101 follower.

Each slider's limits are derived from the robot's saved LeRobot calibration.
Body joints are displayed in degrees; the gripper uses LeRobot's 0..100 scale.
The controller repeatedly compares desired positions with measured positions,
so it does not assume that a command produces an equal physical displacement.
"""

import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "/dev/ttyACM0"
ROBOT_ID = "my_follower"
MAX_RELATIVE_TARGET = 5.0       # Keep consistent with the known-working jog script.
MAX_STEP_DEG = 5.0              # Maximum change in one command for rotational joints.
MAX_STEP_GRIPPER = 5.0           # Gripper uses its 0..100 scale, not degrees.
CONTROL_PERIOD_S = 0.12
POSITION_EPSILON_DEG = 0.8
POSITION_EPSILON_GRIPPER = 1.0
PLAYBACK_WAYPOINT_TIMEOUT_S = 30.0
PLAYBACK_SETTLE_S = 0.25

JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def read_pose(robot):
    observation = robot.get_observation()
    return {
        key[:-4]: float(value)
        for key, value in observation.items()
        if key.endswith(".pos")
    }


def calibration_field(entry, field):
    """Support both dict-based and dataclass-based LeRobot calibration entries."""
    if isinstance(entry, dict):
        return int(entry[field])
    return int(getattr(entry, field))


def calibrated_slider_limits(robot):
    """Return calibrated limits in the same units used by get_observation()."""
    calibration = getattr(robot, "calibration", None)
    if not calibration:
        raise RuntimeError(
            "No saved calibration was loaded. Calibrate the follower first, "
            "then rerun this utility."
        )

    limits = {}
    for joint in JOINTS:
        if joint not in calibration:
            raise RuntimeError(f"Calibration is missing joint {joint!r}.")
        entry = calibration[joint]
        raw_min = calibration_field(entry, "range_min")
        raw_max = calibration_field(entry, "range_max")
        if raw_max <= raw_min:
            raise RuntimeError(
                f"Invalid calibration range for {joint}: {raw_min}..{raw_max}"
            )

        if joint == "gripper":
            # LeRobot exposes the gripper in RANGE_0_100 units.
            limits[joint] = (0.0, 100.0)
        else:
            # LeRobot's DEGREES normalization uses the midpoint of the calibrated
            # encoder range as zero and 4095 ticks per full revolution.
            midpoint = (raw_min + raw_max) / 2.0
            lower = (raw_min - midpoint) * 360.0 / 4095.0
            upper = (raw_max - midpoint) * 360.0 / 4095.0
            limits[joint] = (lower, upper)
    return limits


class SO101SliderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SO-101 Servo Slider Controller")
        self.root.minsize(760, 560)

        self.events = queue.Queue()
        self.targets_lock = threading.Lock()
        self.targets = None
        self.stop_event = threading.Event()
        self.connected = False
        self.closing = False
        self.slider_widgets = {}
        self.current_labels = {}
        self.target_labels = {}
        self.slider_vars = {}
        self.limits = None
        self.last_pose = {}
        self.pose_lock = threading.Lock()
        self.trajectory = []
        self.recording = False
        self.playing = False
        self.playback_thread = None
        self._updating_sliders = False

        top = ttk.Frame(root, padding=(12, 10, 12, 4))
        top.pack(fill="x")
        ttk.Label(top, text=f"Port: {PORT}    Robot ID: {ROBOT_ID}").pack(side="left")
        self.status_var = tk.StringVar(value="Connecting to robot…")
        ttk.Label(top, textvariable=self.status_var).pack(side="right")

        self.joints_frame = ttk.Frame(root, padding=(12, 8))
        self.joints_frame.pack(fill="both", expand=True)

        ttk.Label(self.joints_frame, text="Joint", width=17).grid(row=0, column=0, sticky="w")
        ttk.Label(self.joints_frame, text="Slider", width=12).grid(row=0, column=1, sticky="w")
        ttk.Label(self.joints_frame, text="Target / calibrated range", width=27).grid(row=0, column=2, sticky="w")
        ttk.Label(self.joints_frame, text="Measured position", width=22).grid(row=0, column=3, sticky="w")

        self.connecting_note = ttk.Label(
            self.joints_frame,
            text="Loading saved calibration and reading servo positions…",
        )
        self.connecting_note.grid(row=1, column=0, columnspan=3, sticky="w", pady=12)

        bottom = ttk.Frame(root, padding=(12, 4, 12, 12))
        bottom.pack(fill="x")
        self.hold_button = ttk.Button(bottom, text="Hold current positions", command=self.hold_current, state="disabled")
        self.hold_button.pack(side="left", padx=(0, 6))
        self.start_trajectory_button = ttk.Button(
            bottom, text="Start Trajectory", command=self.start_trajectory, state="disabled"
        )
        self.start_trajectory_button.pack(side="left", padx=3)
        self.record_button = ttk.Button(bottom, text="Record", command=self.record_state, state="disabled")
        self.record_button.pack(side="left", padx=3)
        self.play_button = ttk.Button(bottom, text="Play trajectory", command=self.play_trajectory, state="disabled")
        self.play_button.pack(side="left", padx=3)
        self.trajectory_var = tk.StringVar(value="Trajectory: 0 states")
        ttk.Label(bottom, textvariable=self.trajectory_var).pack(side="left", padx=(8, 0))

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.worker = threading.Thread(target=self.robot_worker, daemon=True)
        self.worker.start()
        self.root.after(100, self.process_events)

    def robot_worker(self):
        robot = None
        connected = False
        try:
            config = SO101FollowerConfig(
                port=PORT,
                id=ROBOT_ID,
                max_relative_target=MAX_RELATIVE_TARGET,
            )
            robot = SO101Follower(config)
            robot.connect(calibrate=False)
            connected = True

            limits = calibrated_slider_limits(robot)
            pose = read_pose(robot)
            missing = [joint for joint in JOINTS if joint not in pose]
            if missing:
                raise RuntimeError(f"Missing position feedback for joints: {missing}")

            with self.targets_lock:
                self.targets = {joint: pose[joint] for joint in JOINTS}
            self.events.put(("connected", {"limits": limits, "pose": pose}))

            while not self.stop_event.is_set():
                cycle_started = time.monotonic()
                pose = read_pose(robot)
                with self.targets_lock:
                    desired = dict(self.targets) if self.targets is not None else None

                if desired is not None:
                    command_pose = dict(pose)
                    should_send = False
                    for joint in JOINTS:
                        current = pose[joint]
                        target = desired[joint]
                        max_step = MAX_STEP_GRIPPER if joint == "gripper" else MAX_STEP_DEG
                        epsilon = POSITION_EPSILON_GRIPPER if joint == "gripper" else POSITION_EPSILON_DEG
                        error = target - current
                        if abs(error) > epsilon:
                            step = max(-max_step, min(max_step, error))
                            command_pose[f"{joint}.pos"] = current + step
                            should_send = True
                        else:
                            command_pose[f"{joint}.pos"] = current

                    if should_send:
                        # Use current measured positions for all other joints. For a
                        # gravity-loaded joint, the next command is based on fresh
                        # feedback, not on the previous command's assumed movement.
                        robot.send_action({f"{joint}.pos": command_pose[f"{joint}.pos"] for joint in JOINTS})

                self.events.put(("pose", pose))
                elapsed = time.monotonic() - cycle_started
                self.stop_event.wait(max(0.0, CONTROL_PERIOD_S - elapsed))

        except Exception as exc:
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            if robot is not None and connected:
                try:
                    robot.disconnect()
                except Exception as exc:
                    self.events.put(("error", f"Disconnect warning: {exc}"))
            self.events.put(("disconnected", None))

    def build_sliders(self, limits, pose):
        self.connecting_note.destroy()
        self.limits = limits
        self._updating_sliders = True

        for row, joint in enumerate(JOINTS, start=1):
            low, high = limits[joint]
            unit = "units" if joint == "gripper" else "°"
            ttk.Label(self.joints_frame, text=joint, width=17).grid(row=row, column=0, sticky="w", pady=7)

            target_var = tk.DoubleVar(value=pose[joint])
            self.slider_vars[joint] = target_var
            scale = tk.Scale(
                self.joints_frame,
                variable=target_var,
                from_=low,
                to=high,
                orient="horizontal",
                resolution=1.0 if joint == "gripper" else 0.5,
                showvalue=False,
                length=390,
                sliderlength=18,
                command=lambda value, name=joint: self.on_slider(name, value),
                state="normal",
            )
            # Left is the calibrated minimum; right is the calibrated maximum.
            scale.grid(row=row, column=1, sticky="ew", padx=(0, 10))
            self.slider_widgets[joint] = scale

            target_text = tk.StringVar(value=f"Target: {pose[joint]:.1f} {unit}\nRange: {low:.1f} … {high:.1f} {unit}")
            ttk.Label(self.joints_frame, textvariable=target_text, width=27).grid(row=row, column=2, sticky="w")
            self.target_labels[joint] = target_text

            current_text = tk.StringVar(value=f"{pose[joint]:.2f} {unit}")
            ttk.Label(self.joints_frame, textvariable=current_text, width=22).grid(row=row, column=3, sticky="w")
            self.current_labels[joint] = current_text

        self.joints_frame.columnconfigure(1, weight=1)
        self._updating_sliders = False
        self.connected = True
        self.status_var.set("Connected")
        self.hold_button.configure(state="normal")
        self.start_trajectory_button.configure(state="normal")
        self.record_button.configure(state="normal")
        self.play_button.configure(state="disabled")

    def on_slider(self, joint, value):
        if self._updating_sliders or not self.connected:
            return
        target = float(value)
        low, high = self.limits[joint]
        target = max(low, min(high, target))
        with self.targets_lock:
            if self.targets is not None:
                self.targets[joint] = target
        unit = "units" if joint == "gripper" else "°"
        self.target_labels[joint].set(
            f"Target: {target:.1f} {unit}\nRange: {low:.1f} … {high:.1f} {unit}"
        )

    def _set_target_display(self, pose):
        """Update slider positions and target labels without issuing jog commands."""
        if not self.slider_vars or not self.limits:
            return
        self._updating_sliders = True
        for joint in JOINTS:
            value = float(pose[joint])
            self.slider_vars[joint].set(value)
            low, high = self.limits[joint]
            unit = "units" if joint == "gripper" else "°"
            self.target_labels[joint].set(
                f"Target: {value:.1f} {unit}\nRange: {low:.1f} … {high:.1f} {unit}"
            )
        self._updating_sliders = False

    def start_trajectory(self):
        if not self.connected or self.playing:
            return
        self.trajectory.clear()
        self.recording = True
        self.trajectory_var.set("Trajectory: 0 states (recording)")
        self.status_var.set("Recording enabled — move the arm and press Record at each waypoint")
        self.play_button.configure(state="disabled")

    def record_state(self):
        if not self.connected or not self.recording or not self.last_pose or self.playing:
            if not self.recording:
                messagebox.showinfo(
                    "Trajectory recording", "Click Start Trajectory before recording states.", parent=self.root
                )
            return
        with self.pose_lock:
            state = {joint: float(self.last_pose[joint]) for joint in JOINTS}
        self.trajectory.append(state)
        self.trajectory_var.set(f"Trajectory: {len(self.trajectory)} states (recording)")
        self.status_var.set(f"Recorded state {len(self.trajectory)}")
        self.play_button.configure(state="normal")

    def play_trajectory(self):
        if not self.connected or self.playing:
            return
        if not self.trajectory:
            messagebox.showinfo("Playback", "The trajectory is empty. Record at least one state first.", parent=self.root)
            return
        if not messagebox.askyesno(
            "Play trajectory",
            f"Move the robot through {len(self.trajectory)} recorded states?\n\n"
            "Keep the arm's workspace clear. Playback will start from the first recorded state.",
            parent=self.root,
        ):
            return

        self.playing = True
        self.recording = False
        for widget in self.slider_widgets.values():
            widget.configure(state="disabled")
        self.hold_button.configure(state="disabled")
        self.start_trajectory_button.configure(state="disabled")
        self.record_button.configure(state="disabled")
        self.play_button.configure(state="disabled")
        self.status_var.set("Playing trajectory…")
        states = [dict(state) for state in self.trajectory]
        self.playback_thread = threading.Thread(target=self._playback_worker, args=(states,), daemon=True)
        self.playback_thread.start()

    def _playback_worker(self, states):
        try:
            for index, state in enumerate(states, start=1):
                if self.stop_event.is_set():
                    break
                # The regular robot worker executes these targets in bounded steps.
                with self.targets_lock:
                    if self.targets is None:
                        raise RuntimeError("Robot targets are not initialized")
                    self.targets.update(state)
                self.events.put(("playback_target", {"index": index, "pose": state, "total": len(states)}))

                deadline = time.monotonic() + PLAYBACK_WAYPOINT_TIMEOUT_S
                reached = False
                while not self.stop_event.is_set() and time.monotonic() < deadline:
                    with self.pose_lock:
                        pose = dict(self.last_pose)
                    if pose and all(
                        abs(pose[joint] - state[joint]) <= (
                            POSITION_EPSILON_GRIPPER if joint == "gripper" else POSITION_EPSILON_DEG
                        )
                        for joint in JOINTS
                    ):
                        reached = True
                        break
                    time.sleep(0.1)

                if self.stop_event.is_set():
                    break
                if not reached:
                    self.events.put((
                        "playback_failed",
                        f"Timed out waiting for waypoint {index}/{len(states)}. "
                        "Playback stopped; check for mechanical obstruction or a joint that cannot reach its target.",
                    ))
                    return
                time.sleep(PLAYBACK_SETTLE_S)

            if not self.stop_event.is_set():
                self.events.put(("playback_done", f"Played {len(states)} recorded states."))
        except Exception as exc:
            self.events.put(("playback_failed", f"Playback error: {type(exc).__name__}: {exc}"))

    def _finish_playback_ui(self, status):
        self.playing = False
        if self.connected:
            for widget in self.slider_widgets.values():
                widget.configure(state="normal")
            self.hold_button.configure(state="normal")
            self.start_trajectory_button.configure(state="normal")
            self.record_button.configure(state="normal")
            self.play_button.configure(state="normal" if self.trajectory else "disabled")
        self.status_var.set(status)
        if self.last_pose:
            self._set_target_display(self.last_pose)
            self.hold_current()

    def hold_current(self):
        if not self.connected or not self.last_pose:
            return
        with self.targets_lock:
            if self.targets is not None:
                self.targets.update({joint: self.last_pose[joint] for joint in JOINTS})
        self._updating_sliders = True
        for joint in JOINTS:
            self.slider_vars[joint].set(self.last_pose[joint])
            low, high = self.limits[joint]
            unit = "units" if joint == "gripper" else "°"
            self.target_labels[joint].set(
                f"Target: {self.last_pose[joint]:.1f} {unit}\nRange: {low:.1f} … {high:.1f} {unit}"
            )
        self._updating_sliders = False

    def process_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "connected":
                    self.build_sliders(payload["limits"], payload["pose"])
                    self.last_pose = payload["pose"]
                elif kind == "pose":
                    with self.pose_lock:
                        self.last_pose = dict(payload)
                    for joint in JOINTS:
                        if joint in self.current_labels:
                            unit = "units" if joint == "gripper" else "°"
                            self.current_labels[joint].set(f"{payload[joint]:.2f} {unit}")
                elif kind == "playback_target":
                    self._set_target_display(payload["pose"])
                    self.status_var.set(f"Playing waypoint {payload['index']}/{payload['total']}…")
                elif kind == "playback_done":
                    self._finish_playback_ui(payload)
                elif kind == "playback_failed":
                    self._finish_playback_ui("Playback stopped")
                    messagebox.showwarning("Trajectory playback", payload, parent=self.root)
                elif kind == "error":
                    self.status_var.set("Error")
                    messagebox.showerror("SO-101 controller error", payload, parent=self.root)
                elif kind == "disconnected":
                    self.connected = False
                    if not self.closing:
                        self.status_var.set("Disconnected")
                        self.hold_button.configure(state="disabled")
                        self.start_trajectory_button.configure(state="disabled")
                        self.record_button.configure(state="disabled")
                        self.play_button.configure(state="disabled")
        except queue.Empty:
            pass

        if not self.closing:
            self.root.after(100, self.process_events)

    def close(self):
        self.closing = True
        self.status_var.set("Disconnecting…")
        self.stop_event.set()
        self.root.after(150, self.finish_close)

    def finish_close(self):
        playback_alive = self.playback_thread is not None and self.playback_thread.is_alive()
        if self.worker.is_alive() or playback_alive:
            self.root.after(150, self.finish_close)
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    SO101SliderGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()