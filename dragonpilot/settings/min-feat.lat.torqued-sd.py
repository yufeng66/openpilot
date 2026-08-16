from dragonpilot.settings import tr

ITEMS = [
  {
    "section": "Lateral",
    "key": "dp_lat_torqued_sd",
    "type": "toggle_item",
    "title": lambda: tr("Speed-Dependent Torque Learner"),
    "description": lambda: tr("Learn latAccelFactor/friction per speed bin and interpolate them by speed, not one global value. Torque-tuned cars only."),
    "brands": ["toyota", "hyundai", "honda", "volkswagen", "rivian"],
    "needs_restart": True,
    "flags": "PERSISTENT",
    "param_type": "BOOL",
    "default": "0",
  },
]
