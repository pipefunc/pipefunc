// graphvizSvg.test.js
import $ from "jquery";
import GraphvizSvg from "./GraphvizSvg";
import "jquery-mousewheel";
import "jquery-color";
import "bootstrap";

describe("GraphvizSvg", () => {
  let container;

  beforeEach(() => {
    document.body.innerHTML = '<div id="graph"></div>';
    container = $("#graph");
  });

  test("should initialize with SVG content", (done) => {
    const svgContent = `<svg width="100pt" height="100pt">
      <g>
        <polygon points="0,0 0,100 100,100 100,0" fill="#ffffff"/>
        <g class="node">
          <title>Node1</title>
          <ellipse cx="50" cy="50" rx="30" ry="30" fill="#ff0000"/>
        </g>
        <g class="edge">
          <title>Edge1</title>
          <path d="M50,50 L70,70" stroke="#000000"/>
        </g>
      </g>
    </svg>`;

    const options = {
      svg: svgContent,
      ready() {
        expect(this.$element.find("svg").length).toBe(1);
        expect(this.$nodes.length).toBe(1);
        expect(this.$edges.length).toBe(1);
        done();
      },
    };

    container.graphviz(options);
  });

  test("should correctly find linked nodes", (done) => {
    const svgContent = `<svg width="100pt" height="100pt">
      <g>
        <g class="node">
          <title>A</title>
          <ellipse cx="50" cy="50" rx="30" ry="30" fill="#ff0000"/>
        </g>
        <g class="node">
          <title>B</title>
          <ellipse cx="150" cy="50" rx="30" ry="30" fill="#00ff00"/>
        </g>
        <g class="edge">
          <title>A->B</title>
          <path d="M50,50 L150,50" stroke="#000000"/>
        </g>
      </g>
    </svg>`;

    const options = {
      svg: svgContent,
      ready() {
        const nodeA = this._nodesByName["A"];
        const linkedNodes = this.linkedFrom(nodeA, false);
        expect(linkedNodes.length).toBe(1);
        expect($(linkedNodes[0]).attr("data-name")).toBe("B");
        done();
      },
    };

    container.graphviz(options);
  });

  // Additional tests for highlight, tooltip, zoom, etc.
});
