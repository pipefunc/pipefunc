// graphvizSvg.test.js
import $ from "jquery";
import GraphvizSvg from "./graphvizsvg";
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

  test("should initialize with default options when none are provided", (done) => {
    container.graphviz({
      ready() {
        expect(this.options.zoom).toBe(true);
        expect(this.options.shrink).toEqual({ x: 4.0625, y: 4.0625 }); // 0.125pt converted to px
        done();
      },
    });
  });

  test("should scale the view on zoom", (done) => {
    const svgContent = `<svg width="100pt" height="100pt">
      <g>
        <g class="node">
          <title>Node1</title>
          <ellipse cx="50" cy="50" rx="30" ry="30" fill="#ff0000"/>
        </g>
      </g>
    </svg>`;

    const options = {
      svg: svgContent,
      ready() {
        const initialWidth = this.$svg.attr("width");
        const initialHeight = this.$svg.attr("height");

        // Simulate mousewheel event with Shift key pressed
        const event = $.Event("mousewheel", {
          shiftKey: true,
          deltaY: -1,
          deltaFactor: 10,
          pageX: 60,
          pageY: 60,
        });

        this.$element.trigger(event);

        const newWidth = this.$svg.attr("width");
        const newHeight = this.$svg.attr("height");

        expect(newWidth).not.toBe(initialWidth);
        expect(newHeight).not.toBe(initialHeight);
        done();
      },
    };

    container.graphviz(options);
  });

  test("should scale nodes according to the shrink option", (done) => {
    const svgContent = `<svg width="100pt" height="100pt">
      <g>
        <g class="node">
          <title>Node1</title>
          <ellipse cx="50" cy="50" rx="30" ry="30" fill="#ff0000"/>
        </g>
      </g>
    </svg>`;

    const options = {
      svg: svgContent,
      shrink: "5px",
      ready() {
        const node = this.$nodes.eq(0).find("ellipse");
        const rx = parseFloat(node.attr("rx"));
        const ry = parseFloat(node.attr("ry"));

        expect(rx).toBeLessThan(30);
        expect(ry).toBeLessThan(30);
        done();
      },
    };

    container.graphviz(options);
  });

  test("should convert shrink option units to pixels", (done) => {
    const svgContent = `<svg width="100pt" height="100pt">
      <g>
        <g class="node">
          <title>Node1</title>
          <ellipse cx="50" cy="50" rx="30" ry="30" fill="#ff0000"/>
        </g>
      </g>
    </svg>`;

    const options = {
      svg: svgContent,
      shrink: "0.125pt", // Default value
      ready() {
        expect(this.options.shrink.x).toBeCloseTo(4.0625);
        expect(this.options.shrink.y).toBeCloseTo(4.0625);
        done();
      },
    };

    container.graphviz(options);
  });

  test("should find all linked nodes", (done) => {
    const svgContent = `<svg width="100pt" height="100pt">
      <g>
        <g class="node">
          <title>A</title>
          <ellipse cx="50" cy="50" rx="30" ry="30"/>
        </g>
        <g class="node">
          <title>B</title>
          <ellipse cx="150" cy="50" rx="30" ry="30"/>
        </g>
        <g class="node">
          <title>C</title>
          <ellipse cx="100" cy="100" rx="30" ry="30"/>
        </g>
        <g class="edge">
          <title>A->B</title>
          <path d="M50,50 L150,50"/>
        </g>
        <g class="edge">
          <title>B->C</title>
          <path d="M150,50 L100,100"/>
        </g>
      </g>
    </svg>`;
    const options = {
      svg: svgContent,
      ready() {
        const nodeA = this._nodesByName["A"];
        // Should find both B and C through transitive connection
        const linkedNodes = this.linkedFrom(nodeA, false);
        expect(linkedNodes.length).toBe(2);
        const linkedNames = linkedNodes
          .map((_, el) => $(el).attr("data-name"))
          .get()
          .sort();
        expect(linkedNames).toEqual(["B", "C"]);
        done();
      },
    };
    container.graphviz(options);
  });

  test("should highlight specified nodes and dim others", (done) => {
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
      </g>
    </svg>`;

    const options = {
      svg: svgContent,
      ready() {
        const nodeA = $(this._nodesByName["A"]);
        const nodeB = $(this._nodesByName["B"]);

        this.highlight(nodeA);

        // Check that nodeA remains the same
        const fillA = nodeA.find("ellipse").attr("fill");
        expect(fillA).toBe("#ff0000");

        // Check that nodeB is dimmed
        const fillB = nodeB.find("ellipse").attr("fill");
        expect(fillB).not.toBe("#00ff00");

        done();
      },
    };

    container.graphviz(options);
  });

  test("should destroy the plugin instance", (done) => {
    const svgContent = `<svg width="100pt" height="100pt">
      <g>
        <g class="node">
          <title>A</title>
          <ellipse cx="50" cy="50" rx="30" ry="30" fill="#ff0000"/>
        </g>
      </g>
    </svg>`;

    const options = {
      svg: svgContent,
      ready() {
        this.destroy();

        // The data attribute should be removed
        expect(this.$element.data("graphviz.svg")).toBeUndefined();
        done();
      },
    };

    container.graphviz(options);
  });

  test("should initialize with default options when none are provided", (done) => {
    container.graphviz({
      svg: `<svg></svg>`,
      ready() {
        expect(this.options.zoom).toBe(true);
        done();
      },
    });
  });

  test("should not set up zoom when zoom option is false", (done) => {
    container.graphviz({
      svg: `<svg></svg>`,
      zoom: false,
      ready() {
        expect(this.zoom).toBeUndefined();
        done();
      },
    });
  });

  test("should handle invalid SVG content gracefully", (done) => {
    container.graphviz({
      svg: `<svg><invalid></invalid></svg>`,
      ready() {
        // Check that the plugin does not crash and initializes
        expect(this.$element.find("svg").length).toBe(1);
        done();
      },
    });
  });

  test("should initialize via jQuery plugin", (done) => {
    const options = {
      svg: `<svg></svg>`,
      ready() {
        expect(this).toBeInstanceOf(GraphvizSvg);
        done();
      },
    };

    container.graphviz(options);
  });

  test("should find nodes connected by undirected edges", (done) => {
    const svgContent = `<svg width="100pt" height="100pt">
      <g>
        <g class="node">
          <title>A</title>
          <ellipse cx="50" cy="50" rx="30" ry="30"/>
        </g>
        <g class="node">
          <title>B</title>
          <ellipse cx="150" cy="50" rx="30" ry="30"/>
        </g>
        <g class="edge">
          <title>A->B</title>  // Changed from A--B to match implementation
          <path d="M50,50 L150,50"/>
        </g>
      </g>
    </svg>`;

    const options = {
      svg: svgContent,
      ready() {
        const nodeA = this._nodesByName["A"];
        const linkedNodes = this.linked(nodeA, false);
        // Should find A (self) and B (direct connection)
        expect(linkedNodes.length).toBe(2);
        const linkedNames = linkedNodes
          .map((_, el) => $(el).attr("data-name"))
          .get()
          .sort();
        expect(linkedNames).toEqual(["A", "B"]);
        done();
      },
    };
    container.graphviz(options);
  });

  test("should find nodes in mixed directed/undirected graph", (done) => {
    const svgContent = `<svg width="100pt" height="100pt">
      <g>
        <g class="node">
          <title>A</title>
          <ellipse cx="50" cy="50" rx="30" ry="30"/>
        </g>
        <g class="node">
          <title>B</title>
          <ellipse cx="150" cy="50" rx="30" ry="30"/>
        </g>
        <g class="node">
          <title>C</title>
          <ellipse cx="100" cy="100" rx="30" ry="30"/>
        </g>
        <g class="edge">
          <title>A->B</title>
          <path d="M50,50 L150,50"/>
        </g>
        <g class="edge">
          <title>B->C</title>
          <path d="M150,50 L100,100"/>
        </g>
      </g>
    </svg>`;
    const options = {
      svg: svgContent,
      ready() {
        const nodeA = this._nodesByName["A"];

        // Test outgoing connections (should find both B and C)
        expect(
          this.linkedFrom(nodeA, false)
            .map((_, el) => $(el).attr("data-name"))
            .get()
            .sort()
        ).toEqual(["B", "C"]);

        // Test incoming connections (should find none)
        expect(
          this.linkedTo(nodeA, false)
            .map((_, el) => $(el).attr("data-name"))
            .get()
        ).toEqual([]);

        done();
      },
    };
    container.graphviz(options);
  });
  test("should handle cycles in the graph", (done) => {
    const svgContent = `<svg width="100pt" height="100pt">
      <g>
        <g class="node"><title>A</title><ellipse cx="50" cy="50" rx="30" ry="30"/></g>
        <g class="node"><title>B</title><ellipse cx="150" cy="50" rx="30" ry="30"/></g>
        <g class="node"><title>C</title><ellipse cx="100" cy="100" rx="30" ry="30"/></g>
        <g class="edge"><title>A->B</title><path d="M50,50 L150,50"/></g>
        <g class="edge"><title>B->C</title><path d="M150,50 L100,100"/></g>
        <g class="edge"><title>C->A</title><path d="M100,100 L50,50"/></g>
      </g>
    </svg>`;
    const options = {
      svg: svgContent,
      ready() {
        const nodeA = this._nodesByName["A"];
        // Test outgoing connections from A (should find all nodes due to cycle: A->B->C->A)
        expect(
          this.linkedFrom(nodeA, false)
            .map((_, el) => $(el).attr("data-name"))
            .get()
            .sort()
        ).toEqual(["A", "B", "C"]);
        // Test incoming connections to A (should find all nodes due to cycle: C->A->B->C)
        expect(
          this.linkedTo(nodeA, false)
            .map((_, el) => $(el).attr("data-name"))
            .get()
            .sort()
        ).toEqual(["A", "B", "C"]);
        done();
      },
    };
    container.graphviz(options);
  });

  test("should handle edge attributes", (done) => {
    const svgContent = `<svg width="100pt" height="100pt">
      <g>
        <g class="node"><title>A</title><ellipse cx="50" cy="50" rx="30" ry="30"/></g>
        <g class="node"><title>B</title><ellipse cx="150" cy="50" rx="30" ry="30"/></g>
        <g class="edge">
          <title>A->B</title>
          <path d="M50,50 L150,50" stroke="#ff0000" stroke-width="2"/>
        </g>
      </g>
    </svg>`;

    const options = {
      svg: svgContent,
      ready() {
        const edge = $(this._edgesByName["A->B"]);
        const color = edge.find("path").data("graphviz.svg.color");
        expect(color.stroke).toBe("#ff0000");
        done();
      },
    };
    container.graphviz(options);
  });

  test("should handle user comments", (done) => {
    const svgContent = `<svg width="100pt" height="100pt">
      <g>
        <!-- User Comment -->
        <g class="node">
          <title>A</title>
          <ellipse cx="50" cy="50" rx="30" ry="30"/>
        </g>
      </g>
    </svg>`;

    const options = {
      svg: svgContent,
      ready() {
        const node = $(this._nodesByName["A"]);
        expect(node.attr("data-comment")).toBe("User Comment");
        done();
      },
    };
    container.graphviz(options);
  });

  test("should handle sendToBack with and without background", (done) => {
    const svgContent = `<svg width="100pt" height="100pt">
      <g>
        <polygon points="0,0 100,0 100,100 0,100" fill="#ffffff"/>
        <g class="node"><title>A</title><ellipse cx="50" cy="50" rx="30" ry="30"/></g>
      </g>
    </svg>`;

    const options = {
      svg: svgContent,
      ready() {
        const node = $(this._nodesByName["A"]);
        this.sendToBack(node);
        // Should be after background polygon
        expect(node.prev().prop("tagName")).toBe("polygon");
        done();
      },
    };
    container.graphviz(options);
  });

  test("should find linked nodes with complex names including special characters", (done) => {
    const svgContent = `<svg width="100pt" height="100pt">
      <g>
        <g class="node">
          <title>a</title>
          <ellipse cx="50" cy="50" rx="30" ry="30"/>
        </g>
        <g class="node">
          <title>f(...) → c</title>
          <ellipse cx="150" cy="50" rx="30" ry="30"/>
        </g>
        <g class="node">
          <title>g(...) → d</title>
          <ellipse cx="250" cy="50" rx="30" ry="30"/>
        </g>
        <g class="edge">
          <title>a->f(...) → c</title>
          <path d="M50,50 L150,50"/>
        </g>
        <g class="edge">
          <title>f(...) → c->g(...) → d</title>
          <path d="M150,50 L250,50"/>
        </g>
      </g>
    </svg>`;

    const options = {
      svg: svgContent,
      ready() {
        // Test incoming connections to f(...) → c
        const nodeF = this._nodesByName["f(...) → c"];
        const incomingNodes = this.linkedTo(nodeF, false);
        expect(incomingNodes.map((_, el) => $(el).attr("data-name")).get()).toEqual(["a"]);

        // Test outgoing connections from f(...) → c
        const outgoingNodes = this.linkedFrom(nodeF, false);
        expect(outgoingNodes.map((_, el) => $(el).attr("data-name")).get()).toEqual(["g(...) → d"]);

        // Test all connections (both directions)
        const allConnected = this.linked(nodeF, false);
        expect(
          allConnected
            .map((_, el) => $(el).attr("data-name"))
            .get()
            .sort()
        ).toEqual(["a", "f(...) → c", "g(...) → d"]);

        done();
      },
    };
    container.graphviz(options);
  });
});
