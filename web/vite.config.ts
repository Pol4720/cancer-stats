/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Rutas relativas: el mismo build funciona en la raíz, bajo una subruta de GitHub Pages
// o servido por FastAPI en el modo en vivo.
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: {
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  // La vista previa reproduce GitHub Pages: sin API, modo estático.
  preview: { proxy: {} },
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 5000,
  },
  test: {
    environment: "jsdom",
    include: ["tests/unit/**/*.test.{ts,tsx}"],
  },
});
