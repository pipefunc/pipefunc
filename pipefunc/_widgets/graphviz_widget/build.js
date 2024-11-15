// build.js
/*
 * Custom build configuration for esbuild that handles WASM files.
 * Key features:
 * - Uses a custom plugin (wasmPlugin) to handle WASM files
 * - Converts WASM to base64 and embeds it in the bundle
 * - Supports watch mode for development
 * - Uses ESM format for compatibility with Jupyter widgets
 *
 * Note: We need to copy the WASM file to ./static first (via npm run copy-wasm)
 * to have a consistent, reliable path to import from. This is more robust than
 * referencing it directly from nested node_modules.
 *
 * This is inspired by:
 * - https://github.com/evanw/esbuild/issues/408#issuecomment-757555771
 * - https://github.com/magjac/d3-graphviz/tree/v3.0.0-bundled-wasm.1
 */
const esbuild = require("esbuild");
const wasmPlugin = require("./esbuild-wasm-plugin");

const watch = process.argv.includes("--watch");

const buildOptions = {
  entryPoints: ["js/widget.js"],
  bundle: true,
  format: "esm",
  outdir: "static",
  plugins: [wasmPlugin],
  sourcemap: watch ? "inline" : false,
  loader: {
    ".wasm": "binary",
  }
};

if (watch) {
  // Use context for watch mode
  esbuild
    .context(buildOptions)
    .then((context) => {
      context.watch();
      console.log("Watching for changes...");
    })
    .catch(() => process.exit(1));
} else {
  // Regular build
  esbuild.build(buildOptions).catch(() => process.exit(1));
}
