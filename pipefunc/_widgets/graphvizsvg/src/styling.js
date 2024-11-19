export function highlight($nodesEdges, tooltips, context) {
  const options = context.options;
  const $everything = context.$nodes.add(context.$edges);
  if ($nodesEdges && $nodesEdges.length > 0) {
    // Dim all other elements
    $everything.not($nodesEdges).each((_, el) => {
      context.colorElement($(el), options.highlight.unselected);
      context.tooltip($(el));
    });
    $nodesEdges.each((_, el) => {
      context.colorElement($(el), options.highlight.selected);
    });
    if (tooltips) {
      context.tooltip($nodesEdges, true);
    }
  } else {
    // Restore all elements
    $everything.each((_, el) => {
      context.restoreElement($(el));
    });
    context.tooltip($everything);
  }
}
export function colorElement($el, getColor, context) {
  const bg = context.$element.css("background");
  $el.find("polygon, ellipse, path").each((_, elem) => {
    const $context = $(elem);
    const color = $context.data("graphviz.svg.color");

    if (color.fill && color.fill !== "none") {
      const newFill = getColor(color.fill, bg);
      $context.attr("fill", newFill);
    }
    if (color.stroke && color.stroke !== "none") {
      const newStroke = getColor(color.stroke, bg);
      $context.attr("stroke", newStroke);
    }
  });
}
export function restoreElement($el, context) {
  $el.find("polygon, ellipse, path").each((_, elem) => {
    const $context = $(elem);
    const color = $context.data("graphviz.svg.color");
    if (color.fill && color.fill != "none") {
      $context.attr("fill", color.fill);
    }
    if (color.stroke && color.stroke != "none") {
      $context.attr("stroke", color.stroke);
    }
  });
}

export function tooltip($elements, show, context) {
  const options = context.options;
  $elements.each(function () {
    $(context)
      .find("a[title]")
      .each(function () {
        if (show) {
          options.tooltips.show.call(context);
        } else {
          options.tooltips.hide.call(context);
        }
      });
  });
}
export function bringToFront($elements, context) {
  $elements.detach().appendTo(context.$graph);
}
export function sendToBack($elements, context) {
  if (context.$background.length) {
    $elements.insertAfter(context.$background);
  } else {
    $elements.detach().prependTo(context.$graph);
  }
}
