import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// No proxy, no server-side rendering -- this is a pure static bundle that
// talks to the already-deployed FastAPI service over HTTP from the
// browser. VITE_API_BASE_URL is read at *build* time (see src/api.ts) and
// baked into the output, which is how Vite env vars work for a static
// site -- see RENDER_REACT_DEPLOY.md for why that matters on Render.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
  },
});
