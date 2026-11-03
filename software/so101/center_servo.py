"""
Move a single Feetech STS servo (STS3215, as in the SO-101) to its middle position, 2048 of 4096,
before attaching the horn during assembly. Connect ONE servo at a time (after lerobot-setup-motors
has given it its ID).

    python center_servo.py --port /dev/ttyACM0 --id 1          # move servo 1 to the middle and hold it
    python center_servo.py --port /dev/ttyACM0 --id 1 --read   # only print the current position

Uses the Feetech SDK installed with lerobot[feetech] (module scservo_sdk).
STS control table: Torque_Enable = 40, Goal_Position = 42 (2 bytes), Present_Position = 56 (2 bytes).
"""
import argparse
import time

import scservo_sdk as scs

TORQUE_ENABLE, GOAL_POSITION, PRESENT_POSITION, MIDDLE = 40, 42, 56, 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--baudrate", type=int, default=1_000_000)   # what lerobot-setup-motors sets
    ap.add_argument("--read", action="store_true")
    args = ap.parse_args()

    port, packet = scs.PortHandler(args.port), scs.PacketHandler(0)
    if not port.openPort() or not port.setBaudRate(args.baudrate):
        raise SystemExit(f"cannot open {args.port} at {args.baudrate} baud")
    try:
        pos, res, err = packet.read2ByteTxRx(port, args.id, PRESENT_POSITION)
        if res != scs.COMM_SUCCESS:
            raise SystemExit(f"no answer from servo id {args.id}: {packet.getTxRxResult(res)}")
        print(f"servo {args.id}: present position {pos} (middle = {MIDDLE})")
        if args.read:
            return
        packet.write1ByteTxRx(port, args.id, TORQUE_ENABLE, 1)
        packet.write2ByteTxRx(port, args.id, GOAL_POSITION, MIDDLE)
        time.sleep(1.5)
        pos, _, _ = packet.read2ByteTxRx(port, args.id, PRESENT_POSITION)
        print(f"servo {args.id}: now at {pos}; torque stays ON so the shaft holds while you mount the horn")
    finally:
        port.closePort()


if __name__ == "__main__":
    main()