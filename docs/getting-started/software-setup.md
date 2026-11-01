# Software setup

The software stack is evolving. Record exact versions for every experiment rather than assuming that the latest package versions are interchangeable.

## Suggested components

- Linux development workstation
- Python virtual environment
- LeRobot for robot configuration and data/control interfaces
- PyBullet for kinematics and simulation experiments
- Git for versioning scripts and experiment configuration

## Environment record

For a reproducible report, capture:

```bash
python --version
pip show lerobot
pip show pybullet
uname -a
```

Also record the Git commit of any local fork or modified dependency. Keep machine-specific paths and credentials out of committed configuration files.
