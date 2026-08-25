from dragonpilot.settings import tr

ITEMS = [
  {
    "section": "Lateral",
    "key": "dp_lat_jerk_torque",
    "type": "toggle_item",
    "title": lambda: tr("Lateral Jerk Torque Controller"),
    # implicit string concatenation, not "+": xgettext only extracts adjacent
    # literals, so a "+"-joined msgid never reaches the .po files
    "description": lambda: tr("Looks ahead at the planned steering to reduce sudden corrections, so the wheel moves more "  # noqa: ISC002
                              "smoothly through turns. Closes the loop in torque space and drives friction from sustained "  # noqa: ISC002
                              "planned lateral jerk instead of instantaneous error. Torque-tuned cars only. "  # noqa: ISC002
                              "Thanks to @twilsonco for the implementation."),
    "brands": ["toyota", "hyundai", "honda", "volkswagen", "rivian"],
    "needs_restart": True,
    "flags": "PERSISTENT",
    "param_type": "BOOL",
    "default": "0",
  },
]
