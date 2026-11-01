# Hardware overview

## SO-101 follower

The physical platform documented here is an SO-101 follower arm assembled from printed parts and serial bus servos.

The current build uses six STS3215 servos with a 1:345 gear ratio and an FE-URT-2 controller board. The controller has been tested over USB serial (the Linux device name may change, for example between `/dev/ttyACM0` and `/dev/ttyACM1`).

These details describe one experimental build, not a universal SO-101 bill of materials. Record the exact servo variant, gear ratio, controller revision, supply voltage, and cable arrangement when reproducing the setup.

## Power notes

Servo supply voltage and current capability matter under load. A bench supply can reveal voltage sag and approximate current draw, but brief current peaks may be missed by slow telemetry polling. Do not infer that a servo has adequate torque solely from a low displayed current reading.

Before increasing supply voltage, verify the safe operating range of the exact servo and controller hardware.

## Record your build

| Item | Record |
|---|---|
| Servo model and ratio | Exact label / seller specification |
| Controller | Board revision and firmware |
| Power supply | Voltage, current rating, wiring |
| Host | OS, kernel, USB serial device |
| Printed parts | Material, slicer, settings |
