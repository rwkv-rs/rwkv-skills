import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During development the Vite dev server (5173) proxies API calls to the
// FastAPI backend (7860) so the SPA and API share an origin.
export default defineConfig({
  base: "/new-eval/",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:7860",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    // Split the heavy, lazily-used markdown/code/math libs out of the main
    // bundle so the dashboard's first paint stays small.
    chunkSizeWarningLimit: 900,
  },
});
