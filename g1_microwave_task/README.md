# G1 microwave door teleop task starter

This starter package registers two Isaac Lab task IDs:

- `Isaac-G1-Open-Microwave-v0`
- `Isaac-G1-Open-Microwave-Play-v0`

## What you still need to edit

Open `g1_microwave_env_cfg.py` and update these constants first:

- `MICROWAVE_USD_PATH`
- `MICROWAVE_DOOR_JOINT`
- `SUCCESS_OPEN_THRESHOLD`
- `TABLE_POS`
- `MICROWAVE_POS`
- `MICROWAVE_ROT_WXYZ`

## Run teleop

```bash
./isaaclab.sh -p /path/to/teleop_g1_microwave.py \
  --task Isaac-G1-Open-Microwave-Play-v0 \
  --teleop_device handtracking \
  --device cuda:0 \
  --enable_pinocchio
```

For Quest motion controllers instead of hand tracking:

```bash
./isaaclab.sh -p /path/to/teleop_g1_microwave.py \
  --task Isaac-G1-Open-Microwave-Play-v0 \
  --teleop_device motion_controllers \
  --device cuda:0 \
  --enable_pinocchio
```

## Record demonstrations

```bash
./isaaclab.sh -p /path/to/record_g1_microwave_demos.py \
  --task Isaac-G1-Open-Microwave-Play-v0 \
  --teleop_device handtracking \
  --dataset_file ./datasets/g1_microwave_demos.hdf5 \
  --num_demos 20 \
  --device cuda:0 \
  --enable_pinocchio
```

## Notes

- This version fixes the G1 base by default for a more stable first dataset.
- Success is currently defined only by the microwave door joint crossing `SUCCESS_OPEN_THRESHOLD`.
- If your microwave opens in the negative joint direction, flip the threshold logic or use a negative threshold.
