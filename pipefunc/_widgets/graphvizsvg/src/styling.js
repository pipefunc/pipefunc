// styling.js
import $ from "jquery";

export function highlight($nodesEdges, tooltips, context) {
  const options = context.options;
  const $everything = context.$nodes.add(context.$edges);
  if ($nodesEdges && $nodesEdges.length > 0) {
    // Dim all other elements
    $everything.not($nodesEdges).each((_, el) => {
      colorElement($(el), options.highlight.unselected, context);
      tooltip($(el), false, context);
    });
    $nodesEdges.each((_, el) => {
      colorElement($(el), options.highlight.selected, context);
    });
    if (tooltips) {
      tooltip($nodesEdges, true, context);
    }
  } else {
    // Restore all elements
    $everything.each((_, el) => {
      restoreElement($(el), context);
    });
    tooltip($everything, false, context);
  }
}


export function colorElement($el, getColor, context) {
  const bg = context.$element.css("background");
  $el.find("polygon, ellipse, path").each((_, elem) => {
    const $element = $(elem);
    const color = $element.data("graphviz.svg.color");

    if (color.fill && color.fill !== "none") {
      const newFill = getColor(color.fill, bg);
      $element.attr("fill", newFill);
    }
    if (color.stroke && color.stroke !== "none") {
      const newStroke = getColor(color.stroke, bg);
      $element.attr("stroke", newStroke);
    }
  });
}

export function restoreElement($el, context) {
  $el.find("polygon, ellipse, path").each((_, elem) => {
    const $element = $(elem);
    const color = $element.data("graphviz.svg.color");
    if (color.fill && color.fill != "none") {
      $element.attr("fill", color.fill);
    }
    if (color.stroke && color.stroke != "none") {
      $element.attr("stroke", color.stroke);
    }
  });
}

export function tooltip($elements, show, context) {
  const options = context.options;
  $elements.each(function () {
    $(this)
      .find("a[title]")
      .each(function () {
        if (show) {
          options.tooltips.show.call(this);
        } else {
          options.tooltips.hide.call(this);
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
