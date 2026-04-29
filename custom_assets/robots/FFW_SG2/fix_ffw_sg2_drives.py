"""One-shot script to repair the USD drive parameters of FFW_SG2.usd.

Usage:
    LD_LIBRARY_PATH=$PXR_BASE/bin:$KIT_PLUGINS:$ISAAC_LIB:$LD_LIBRARY_PATH \
    PYTHONPATH=$PXR_BASE \
    python custom_assets/robots/ffw_sg2/fix_ffw_sg2_drives.py

Or just run inside Isaac Sim's Script Editor (Window -> Script Editor).
"""

from __future__ import annotations

from pxr import Usd

USD_PATH = "custom_assets/robots/ffw_sg2/FFW_SG2.usd"
ROOT = "/Root/ffw_sg2_follower/joints"

WHEELS = [
    "left_wheel_steer",
    "left_wheel_drive",
    "right_wheel_steer",
    "right_wheel_drive",
    "rear_wheel_steer",
    "rear_wheel_drive",
]

WHEEL_DRIVE = {
    "drive:angular:physics:targetPosition": 0.0,
    "drive:angular:physics:targetVelocity": 0.0,
    "drive:angular:physics:stiffness": 500.0,
    "drive:angular:physics:damping": 50.0,
    "drive:angular:physics:maxForce": 200.0,
}

LIFT_DRIVE = {
    "drive:linear:physics:targetPosition": -0.4,
    "drive:linear:physics:targetVelocity": 0.0,
    "drive:linear:physics:stiffness": 2000.0,
    "drive:linear:physics:damping": 200.0,
    "drive:linear:physics:maxForce": 2000.0,
}


def set_attrs(prim: Usd.Prim, attrs: dict[str, float]) -> None:
    for name, value in attrs.items():
        attr = prim.GetAttribute(name)
        if not attr:
            print(f"  ! attribute missing: {name}")
            continue
        old = attr.Get()
        attr.Set(value)
        print(f"  {name}: {old} -> {value}")


def main() -> None:
    stage = Usd.Stage.Open(USD_PATH)
    if not stage:
        raise RuntimeError(f"failed to open {USD_PATH}")

    for wheel in WHEELS:
        path = f"{ROOT}/{wheel}"
        prim = stage.GetPrimAtPath(path)
        if not prim:
            print(f"missing prim: {path}")
            continue
        print(f"[wheel] {path}")
        set_attrs(prim, WHEEL_DRIVE)

    lift_path = f"{ROOT}/lift_joint"
    lift = stage.GetPrimAtPath(lift_path)
    if lift:
        print(f"[lift] {lift_path}")
        set_attrs(lift, LIFT_DRIVE)

    stage.GetRootLayer().Save()
    print("saved", USD_PATH)


if __name__ == "__main__":
    main()
