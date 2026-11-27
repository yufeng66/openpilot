"""dp - compact lead readout ("Rel+Gap" mode of dp_ui_lead): one line under the lead marker, for example "-5, 1.8".

The first number is the lead's speed minus ours (mph, km/h if metric), the second the time gap dRel / vEgo in
seconds. Same text as sunnypilot's chevron metrics REL_SPEED_GAP option. Pure functions, no raylib, so they can be
tested without a display.
"""
import math

from openpilot.common.constants import CV

FONT_SIZE = 64
LINE_HEIGHT = int(FONT_SIZE * 1.25)
MARGIN = 20
GAP_BELOW_CHEVRON = 15
ALPHA_MATCHED = 1.0  # the lead is matched to a radar point (lead.radar)
ALPHA_VISION = 0.55  # the lead comes from the camera model only
SHADOW_ALPHA = 200
# Width kept free at the left edge of the HUD rect. It mirrors the left column of selfdrive/ui/onroad/hud_renderer.py:
# a three-digit current speed in Inter Bold at FONT_SIZES.current_speed (176 px) starts at LEFT_COLUMN_MIN_X (16 px)
# and ends about 345 px from the rect edge for 100-199 (measured on a rendered frame), which is wider than the MAX
# box (ends at 260 px). Update it if that column changes.
LEFT_COLUMN_RESERVED_PX = 360


def rel_gap_text(d_rel: float, v_rel: float, v_ego: float, is_metric: bool) -> str:
  """'<lead speed minus ours, signed integer>, <time gap in s, one decimal>' with no units; '0', not '-0'.
  A part whose input is not finite reads '---' (this runs in the UI loop and must never raise)."""
  multiplier = CV.MS_TO_KPH if is_metric else CV.MS_TO_MPH
  rel_text = "---"
  if math.isfinite(v_rel):
    rel = int(round(v_rel * multiplier))
    rel_text = f"{rel:+d}" if rel != 0 else "0"

  gap = 0.0
  if math.isfinite(d_rel) and math.isfinite(v_ego) and d_rel > 0 and v_ego > 0:
    gap = d_rel / v_ego
  gap_text = f"{gap:.1f}" if (0 < gap < 200) else "---"
  return f"{rel_text}, {gap_text}"


def text_alpha(radar: bool) -> float:
  return ALPHA_MATCHED if radar else ALPHA_VISION


def rel_gap_text_position(chevron_x: float, chevron_y: float, sz: float, text_width: float,
                          rect_x: float, rect_y: float, rect_width: float, rect_height: float,
                          left_reserved: float = LEFT_COLUMN_RESERVED_PX) -> tuple[float, float]:
  """Top-left corner of the line: centred under the chevron tip, kept inside the rect with a margin, right of the
  left_reserved strip (the HUD's speed / RPM column), and above the chevron when there is no room below it."""
  bottom = rect_y + rect_height - MARGIN
  y = chevron_y + sz + GAP_BELOW_CHEVRON
  if y + LINE_HEIGHT > bottom:
    y = min(chevron_y, bottom) - GAP_BELOW_CHEVRON - LINE_HEIGHT
    y = max(rect_y + MARGIN, y)

  x_min = rect_x + max(MARGIN, left_reserved + MARGIN)
  x_max = rect_x + rect_width - MARGIN - text_width
  x = min(max(chevron_x - text_width / 2, x_min), x_max)
  return max(x, x_min), y  # a text too wide for the space left starts right of the column anyway
