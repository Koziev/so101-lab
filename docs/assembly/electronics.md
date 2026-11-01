# Electronics and wiring

Document the controller, servo bus, power distribution, and host connection here.

## Current experimental setup

The follower build has been tested with an FE-URT-2 controller and six STS3215 serial bus servos. The controller has appeared as a USB serial device such as `/dev/ttyACM0` or `/dev/ttyACM1`; device names can change after reconnecting hardware.

## Wiring documentation checklist

- Controller board revision and pinout.
- Servo bus topology and connector orientation.
- Power supply voltage, current rating, and polarity.
- Grounding and USB connection.
- Emergency power-disconnect procedure.

!!! danger "Power safety"
    Confirm polarity and voltage limits from the documentation for the exact board and servo variant before connecting power. Disconnect power before changing wiring.
