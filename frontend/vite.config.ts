import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const API = "http://127.0.0.1:8088";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The dev server proxies to the Python process so the UI has one origin
    // in development and in production alike.
    proxy: {
      "/api": { target: API, changeOrigin: true, ws: false },
    },
  },
  build: { outDir: "dist", sourcemap: true },
});
