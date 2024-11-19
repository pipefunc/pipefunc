// utils.ts
export class ColorUtil {
  static parseColor(color) {
    // Handle rgb format (for background)
    if (color.startsWith("rgb")) {
      const match = color.match(/^rgb\((\d+),\s*(\d+),\s*(\d+)/);
      if (match) {
        return {
          r: parseInt(match[1], 10),
          g: parseInt(match[2], 10),
          b: parseInt(match[3], 10),
        };
      }
    }

    // Handle hex format (from Graphviz)
    const h = color.replace("#", "");
    return {
      r: parseInt(h.substring(0, 2), 16),
      g: parseInt(h.substring(2, 4), 16),
      b: parseInt(h.substring(4, 6), 16),
    };
  }

  static transition(color1, color2, amount) {
    const c1 = ColorUtil.parseColor(color1);
    const c2 = ColorUtil.parseColor(color2.match(/^rgb\([^)]+\)/)?.[0] ?? color2);

    const r = Math.round(c1.r + (c2.r - c1.r) * amount);
    const g = Math.round(c1.g + (c2.g - c1.g) * amount);
    const b = Math.round(c1.b + (c2.b - c1.b) * amount);
    const r_str = r.toString(16).padStart(2, "0");
    const g_str = g.toString(16).padStart(2, "0");
    const b_str = b.toString(16).padStart(2, "0");
    return `#${r_str}${g_str}${b_str}`;
  }
}

export function convertToPx(val, gv_pt_2_px) {
  let retval = val;
  if (typeof val === "string") {
    let end = val.length;
    let factor = 1.0;
    if (val.endsWith("px")) {
      end -= 2;
    } else if (val.endsWith("pt")) {
      end -= 2;
      factor = gv_pt_2_px;
    }
    retval = parseFloat(val.substring(0, end)) * factor;
  }
  return retval;
}
