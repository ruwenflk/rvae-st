import numpy as np


def lenze_without_rot_angle_and_spd(df):
    df.drop(labels=["t"], axis=1, inplace=True, errors="ignore")
    df.drop(
        labels=["lstdrehmoment_act_motor_torq"], axis=1, inplace=True, errors="ignore"
    )
    df.drop(labels=["U32_rot_angle_rotor"], axis=1, inplace=True, errors="ignore")
    df.drop(labels=["S32_act_rot_spd"], axis=1, inplace=True, errors="ignore")
    return df.astype(np.float32)


def ett(df):
    df.drop(labels=["date"], axis=1, inplace=True)
    return df.astype(np.float32)


def ecg_heart(df):
    df = df.iloc[:555556]
    df = df[["ECG1", "ECG2"]]
    return df.astype(np.float32)
