from .heuristics import MiddleDistCentroid,MiddleWingAngleMinIntruder,MiddleWingAngleMaxIntruder,MiddleAxisRatioResident,MiddleAxisRatioIntruder, MiddleAngleSocialIntruder, MiddleSpeedResident, MiddleWingAngleMaxResident, MiddleSpeedIntruder,MiddleWingAngleMinResident, MiddleAngularSpeedResident, MiddleAngularSpeedIntruder, MiddleAngleSocial,RandomLabeler

# USAGE: import label function and add to this list
label_functions_list = [
    MiddleDistCentroid,
    MiddleSpeedResident,
    MiddleSpeedIntruder,
    MiddleAngularSpeedResident,
    MiddleAngularSpeedIntruder,
    MiddleAngleSocial,
    MiddleWingAngleMinResident,
    MiddleWingAngleMaxResident,
    MiddleWingAngleMinIntruder,
    MiddleWingAngleMaxIntruder,    
    MiddleAngleSocialIntruder,
    MiddleAxisRatioResident,
    MiddleAxisRatioIntruder,
    RandomLabeler
]
