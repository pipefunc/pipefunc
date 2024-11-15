# Graphviz Widget

## Build Process
The build process involves several steps to properly handle the WASM binary:

1. `npm run copy-wasm`: Copies graphvizlib.wasm from d3-graphviz's dependencies to ./static
   - This provides a consistent path for importing the WASM file
   - More reliable than referencing nested node_modules directly
   - Used by both build and dev commands

2. `npm run build`: Production build
   - Runs copy-wasm first
   - Uses esbuild with custom WASM plugin
   - Embeds WASM binary in the final bundle

3. `npm run dev`: Development mode
   - Runs copy-wasm first
   - Same build process as production
   - Adds watch mode and source maps for development
   - Auto-rebuilds when files change

## Implementation Notes
- The WASM binary is embedded in the JavaScript bundle as base64
- We override the fetch API to intercept WASM file requests
- Web Worker mode is disabled to ensure consistent WASM loading
- This approach works in any Jupyter environment without needing a separate file server

## Dependencies
- d3-graphviz: Provides the core graphviz functionality
- The WASM binary comes from @hpcc-js/wasm (via d3-graphviz)
