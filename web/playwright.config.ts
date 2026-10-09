import { defineConfig } from "@playwright/test";

// Pruebas de extremo a extremo sobre el build estático (vite preview).
export default defineConfig({
  testDir: "tests/e2e",
  timeout: 60_000,
  use: {
    baseURL: "http://127.0.0.1:4173",
    launchOptions: process.env.PW_CHROMIUM ? { executablePath: process.env.PW_CHROMIUM } : {},
  },
  webServer: {
    command: "npx vite preview --host 127.0.0.1 --port 4173 --strictPort",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: !process.env.CI,
  },
  projects: [
    { name: "escritorio", use: { viewport: { width: 1366, height: 860 } } },
    { name: "movil", use: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true } },
  ],
});
