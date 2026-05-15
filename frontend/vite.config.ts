import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8765", changeOrigin: true },
      // When using only Vite (no built dist), iframe assets can be proxied to the same API server.
      "/training-yaml-ui": { target: "http://127.0.0.1:8765", changeOrigin: true },
    },
  },
});
