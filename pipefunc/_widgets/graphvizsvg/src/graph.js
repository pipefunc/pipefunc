// graph.js
import $ from "jquery";

export function linkedTo(node, includeEdges, context) {
  const $retval = $();
  findLinked(
    node,
    includeEdges,
    (nodeName, edgeName) => {
      let other = null;
      const connection = edgeName.split("->");
      if (
        connection.length > 1 &&
        (connection[1] === nodeName || connection[1].startsWith(nodeName + ":"))
      ) {
        return connection[0].split(":")[0];
      }
      return other;
    },
    $retval,
    context
  );
  return $retval;
}

export function linkedFrom(node, includeEdges, context) {
  const $retval = $();
  findLinked(
    node,
    includeEdges,
    (nodeName, edgeName) => {
      let other = null;
      const connection = edgeName.split("->");
      if (
        connection.length > 1 &&
        (connection[0] === nodeName || connection[0].startsWith(nodeName + ":"))
      ) {
        return connection[1].split(":")[0];
      }
      return other;
    },
    $retval,
    context
  );
  return $retval;
}

export function linked(node, includeEdges, context) {
  const $retval = $();
  $retval.push(node); // Add the original node
  const fromNodes = linkedFrom(node, includeEdges, context);
  const toNodes = linkedTo(node, includeEdges, context);
  return $retval.add(fromNodes).add(toNodes);
}

export function findEdge(nodeName, testEdge, $retval, context) {
  const retval = [];
  for (const name in context._edgesByName) {
    const match = testEdge(nodeName, name);
    if (match) {
      if ($retval) {
        $retval.push(context._edgesByName[name]);
      }
      retval.push(match);
    }
  }
  return retval;
}

export function findLinked(node, includeEdges, testEdge, $retval, context) {
  const $node = $(node);
  let $edges = null;
  if (includeEdges) {
    $edges = $retval;
  }
  const names = findEdge($node.attr("data-name"), testEdge, $edges, context);
  names.forEach((name) => {
    const n = context._nodesByName[name];
    if (!$retval.is(n)) {
      $retval.push(n);
      findLinked(n, includeEdges, testEdge, $retval, context);
    }
  });
}
