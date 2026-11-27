from dragonpilot.settings import tr

ITEMS = [
  {
    "section": "Honda",
    "key": "dp_honda_bosch_radar",
    "type": "toggle_item",
    "title": lambda: tr("Use Camera Objects for the Lead Car (Experimental)"),
    # implicit string concatenation, not "+": xgettext only extracts adjacent literals
    "description": lambda: tr("Use the list of objects that the stock Honda camera sends to the radar unit to measure the distance and speed of the lead car. "  # noqa: ISC002
                              "The car's own cruise control still controls gas and brake, but openpilot uses this lead "  # noqa: ISC002
                              "to decide when to press Resume after a stop and when to show its collision warning. "  # noqa: ISC002
                              "Accord 2018-22 with stock ACC only; off when openpilot controls gas and brake. "  # noqa: ISC002
                              "Experimental. Applies at the next start of the car."),
    "brands": ["honda"],
    "needs_restart": True,
    "flags": "PERSISTENT",
    "param_type": "BOOL",
    "default": "0",
  },
]
