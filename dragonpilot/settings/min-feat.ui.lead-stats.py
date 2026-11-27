from dragonpilot.settings import tr

ITEMS = [
  {
    "section": "UI",
    "key": "dp_ui_lead",
    "type": "text_spin_button_item",
    "title": lambda: tr("Display Lead Stats"),
    # dp - Rel+Gap; implicit string concatenation, not "+": xgettext only extracts adjacent literals
    "description": lambda: tr("Display the statistics of lead car and/or radar tracking points.<br>Lead: Lead stats only<br>"  # noqa: ISC002
                              "Radar: Radar tracking point stats only<br>All: Lead and Radar stats<br>"  # noqa: ISC002
                              "Rel+Gap: one line under the lead car, its speed minus yours and the time gap in seconds "  # noqa: ISC002
                              "(dimmed when only the camera sees it)<br>NOTE: Radar option only works on certain vehicle models."),
    "options": [lambda: tr("Off"), lambda: tr("Lead"), lambda: tr("Radar"), lambda: tr("All"), lambda: tr("Rel+Gap")],  # dp - Rel+Gap
    "default": "0",
    "condition": "not MICI",
    "flags": "PERSISTENT",
    "param_type": "INT",
  },
]
