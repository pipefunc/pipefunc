// esbuild-wasm-plugin.js
/*
 * Custom esbuild plugin to handle WASM files.
 *
 * This plugin:
 * 1. Intercepts .wasm file imports
 * 2. Reads the WASM file
 * 3. Converts it to base64 string
 * 4. Creates a JavaScript module that converts the base64 back to Uint8Array
 *
 * This approach allows the WASM binary to be embedded directly in the JavaScript
 * bundle, avoiding the need for separate file loading at runtime.
 */
const fs = require("fs");
const path = require("path");

const wasmPlugin = {
  name: "wasm",
  setup(build) {
    build.onResolve({ filter: /\.wasm$/ }, (args) => {
      const resolvedPath = path.join(args.resolveDir, args.path);
      return {
        path: resolvedPath,
        namespace: "wasm-binary",
      };
    });

    build.onLoad({ filter: /.*/, namespace: "wasm-binary" }, async (args) => {
      const buffer = await fs.promises.readFile(args.path);
      // Convert the buffer to base64
      const base64 = buffer.toString("base64");
      const contents = `
        const wasmBase64 = "${base64}";
        const wasmBinary = Uint8Array.from(atob(wasmBase64), c => c.charCodeAt(0));
        export default wasmBinary;
      `;
      return {
        contents,
        loader: "js",
      };
    });
  },
};

module.exports = wasmPlugin;
