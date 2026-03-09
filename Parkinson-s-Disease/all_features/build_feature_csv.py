thisimport os
import pandas as pd
import numpy as np
from pose_extract import extract_leg_joints
from turning_features import extract_turning_features

DATA = []
LABELS = []
VIDEO_NAMES = []

# Training data from actual PD patient videos
ROOT = "../Actual videos"

if not os.path.exists(ROOT):
    print(f"Error: Folder '{ROOT}' not found.")
    exit()

print(f"Scanning folder: {ROOT}")

try:
    for vid in sorted(os.listdir(ROOT)):
        if vid.endswith(".mp4"):
            path = os.path.join(ROOT, vid)
            label = 1  # All "Actual videos" are PD patients
            try:
                joints = extract_leg_joints(path)
                if len(joints) < 30:
                    print(f"Skipping {vid}: Not enough frames")
                    continue
                feats, names = extract_turning_features(joints)
                DATA.append(feats)
                LABELS.append(label)
                VIDEO_NAMES.append(vid)
                print(f"Processed {vid}")
            except Exception as e:
                print(f"Error processing {vid}: {e}")

except KeyboardInterrupt:
    print("\nProcessing interrupted.")

finally:
    if not DATA:
        print("No features extracted.")
        exit()

    print(f"\nExtracted {len(DATA)} real PD samples.")
    print("Building balanced dataset with augmentation & realistic healthy controls...")

    FINAL_DATA = []
    FINAL_LABELS = []
    FINAL_VIDEO_NAMES = []

    np.random.seed(42)

    for i in range(len(DATA)):
        orig = np.array(DATA[i], dtype=float)
        vid = VIDEO_NAMES[i]

        # ── PD AUGMENTATION (10 variations, 5% noise) ──────────────────────
        for j in range(10):
            noise = np.random.normal(1.0, 0.05, len(orig))
            FINAL_DATA.append((orig * noise).tolist())
            FINAL_LABELS.append(1)
            FINAL_VIDEO_NAMES.append(f"PD_{j}_{vid}")

        # ── SYNTHETIC HEALTHY (10 variations, moderate difference + overlap) ─
        for j in range(10):
            h = orig.copy()

            vel_factor      = np.random.uniform(1.3, 1.8)
            dur_factor      = np.random.uniform(0.50, 0.70)
            steps_factor    = np.random.uniform(0.45, 0.65)
            steplen_factor  = np.random.uniform(1.3, 1.6)
            stepvar_factor  = np.random.uniform(0.45, 0.75)
            freeze_factor   = np.random.uniform(0.0, 0.18)
            knee_factor     = np.random.uniform(1.05, 1.20)

            h[1] *= vel_factor
            h[2] *= dur_factor
            h[3]  = max(2, int(h[3] * steps_factor))
            h[4] *= steplen_factor
            h[5] *= stepvar_factor
            h[6]  = max(0.0, orig[6] * freeze_factor)
            h[7] *= knee_factor

            noise = np.random.normal(1.0, 0.08, len(h))
            FINAL_DATA.append((h * noise).tolist())
            FINAL_LABELS.append(0)
            FINAL_VIDEO_NAMES.append(f"CTRL_{j}_{vid}")

    df = pd.DataFrame(FINAL_DATA, columns=names)
    df["label"] = FINAL_LABELS
    df["filename"] = FINAL_VIDEO_NAMES

    pd_count   = sum(1 for l in FINAL_LABELS if l == 1)
    ctrl_count = sum(1 for l in FINAL_LABELS if l == 0)

    df.to_csv("turning_pd_features.csv", index=False)
    print(f"\nDataset saved: {len(df)} total rows  ({pd_count} PD | {ctrl_count} Control)")
    print("Ready to train. Run: python train_model.py")
