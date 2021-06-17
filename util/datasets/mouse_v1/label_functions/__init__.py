from .heuristics import MiddleSpeedResident, MiddleAngleSocial, MiddleAngleSocialIntruder, MiddleSpeedIntruder, MiddleMovementNoseResident, MiddleMovementNoseIntruder, MiddleDistNoseNose, MiddleDistNoseTail, MiddleAngleHeadBodyResident, MiddleAngleHeadBodyIntruder, ReadLabels


# USAGE: import label function and add to this list
label_functions_list = [
    MiddleSpeedResident,
    MiddleSpeedIntruder,
    MiddleAngleSocial,
    MiddleAngleSocialIntruder,
    MiddleMovementNoseResident,
    MiddleMovementNoseIntruder,
    MiddleDistNoseNose,
    MiddleDistNoseTail,
    MiddleAngleHeadBodyResident,
    MiddleAngleHeadBodyIntruder,
    ReadLabels
]
