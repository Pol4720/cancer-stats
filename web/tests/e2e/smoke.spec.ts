import { expect, test } from "@playwright/test";

const SECTIONS = [
  "inicio", "datos", "depuracion", "ausentes", "exploracion", "atipicos", "modelo",
  "final", "diagnostico", "sensibilidad", "prediccion", "corridas", "ejecutar",
];

test("todas las secciones se muestran sin errores ni desbordamiento", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  for (const id of SECTIONS) {
    await page.goto(`/#/${id}`);
    await expect(page.locator(".section-head h1")).toBeVisible({ timeout: 30_000 });
    await expect(page.locator(".error-box")).toHaveCount(0);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow, `desbordamiento en ${id}`).toBeLessThanOrEqual(1);
  }
  expect(errors).toEqual([]);
});

test("el modelo final muestra R² ajustado y RMSE", async ({ page }) => {
  await page.goto("/#/final");
  await expect(page.getByText("R² ajustado").first()).toBeVisible();
  await expect(page.getByText("RMSE").first()).toBeVisible();
  await page.getByRole("button", { name: "Clásicos (SPSS)" }).click();
  await expect(page.getByRole("button", { name: "Clásicos (SPSS)" })).toHaveAttribute("aria-pressed", "true");
});

test("el tema oscuro se aplica y se recuerda", async ({ page, isMobile }) => {
  test.skip(isMobile, "basta con comprobarlo en escritorio");
  await page.goto("/#/inicio");
  await page.getByRole("button", { name: "Tema oscuro" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
});

test("el modo guía avanza con el teclado", async ({ page, isMobile }) => {
  test.skip(isMobile, "navegación por teclado en escritorio");
  await page.goto("/#/inicio");
  await page.getByRole("button", { name: /Modo guía/ }).click();
  await expect(page.getByText("Paso 1 de 13")).toBeVisible();
  await page.keyboard.press("ArrowRight");
  await expect(page).toHaveURL(/#\/datos/);
  await expect(page.getByText("Paso 2 de 13")).toBeVisible();
});

test("el menú lateral se abre en móvil", async ({ page, isMobile }) => {
  test.skip(!isMobile, "sólo en móvil");
  await page.goto("/#/inicio");
  await page.getByRole("button", { name: "Abrir el menú" }).click();
  await page.getByRole("link", { name: /Diagnóstico/ }).click();
  await expect(page).toHaveURL(/#\/diagnostico/);
});
