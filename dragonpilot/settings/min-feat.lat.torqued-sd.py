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
  {
    "section": "Lateral",
    "key": "dp_lat_torqued_sd_friction",
    "type": "spin_button_item",
    "title": lambda: tr("Friction Reduction"),
    # implicit string concatenation, not "+": xgettext only extracts adjacent
    # literals, so a "+"-joined msgid never reaches the .po files
    "description": lambda: tr("Lower the learned friction the controller applies, which can make steering feel smoother. "  # noqa: ISC002
                              "Each step is -10%, and only speed bins the learner has validated are scaled. "  # noqa: ISC002
                              "Read once at engage, so change it with the car off."),
    "brands": ["toyota", "hyundai", "honda", "volkswagen", "rivian"],
    "depends_on": "dp_lat_torqued_sd == 1",
    "needs_restart": True,
    "default": "0",
    "min_val": 0,
    "max_val": 9,
    "step": 1,
    "suffix": lambda: tr(" x -10%"),
    "flags": "PERSISTENT",
    "param_type": "INT",
  },
]
