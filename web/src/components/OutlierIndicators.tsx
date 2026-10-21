import { Card, DataTable, Explain, StatusTag } from "./ui";
import { ci, int, num, pval, signed, termLabel } from "../lib/format";
import { useApp } from "../lib/store";

/** Fila de la tabla de modelos con indicadoras (el modelo final va primero, como referencia). */
function comparisonRows(oi: any): any[] {
  const models = oi.models as any[];
  if (!models.length) return [];
  const f = models[0];
  return [
    { label: "Final (sin indicadoras)", q: 0, r2_adj: f.r2_adj_final, sigma: f.sigma_final, aic: f.aic_final, bic: f.bic_final, final: true },
    ...models,
  ];
}

function Verdict({ m }: { m: any }) {
  if (m.final) return <span className="muted">referencia</span>;
  const yes = m.veredicto.vale_la_pena;
  return <StatusTag status={yes ? "advertencia" : "ok"} label={yes ? "Excluir" : "Mantener"} />;
}

/** Tabla de comparación del modelo final con los modelos con variables indicadoras de atípicos. */
export function IndicatorComparison({ name = "indicadoras" }: { name?: string }) {
  const { res } = useApp();
  const oi = res.effects.outlier_indicators;
  if (!oi) return null;
  return (
    <DataTable
      name={name}
      csv={false}
      rows={comparisonRows(oi)}
      columns={[
        { key: "label", header: "Modelo", wrap: true, render: (r: any) => (r.final ? r.label : r.label.replace("Final + indicadoras de los ", "+ ")) },
        { key: "veredicto", header: "Decisión", value: (r: any) => r.veredicto?.vale_la_pena, render: (r: any) => <Verdict m={r} /> },
        { key: "q", header: "Indicadoras", num: true },
        { key: "r2_adj", header: "R² aj.", num: true, render: (r: any) => num(r.r2_adj, 3) },
        { key: "sigma", header: "σ residual", num: true, render: (r: any) => num(r.sigma, 2) },
        { key: "bic", header: "BIC", num: true, render: (r: any) => num(r.bic, 1) },
        { key: "F", header: "F del bloque", num: true, render: (r: any) => num(r.F, 2) },
        { key: "p", header: "p nominal", num: true, render: (r: any) => (r.final ? "—" : pval(r.p)) },
        { key: "p_calibrado", header: "p calibrado", num: true, render: (r: any) => pval(r.p_calibrado) },
        { key: "max_cambio_se", header: "Máx. |Δβ|/EE", num: true, render: (r: any) => num(r.max_cambio_se, 2) },
        { key: "cambian", header: "Cambian", num: true, value: (r: any) => r.cambian?.length, render: (r: any) => (r.final ? "—" : int(r.cambian.length)) },
        { key: "aic", header: "AIC", num: true, render: (r: any) => num(r.aic, 1) },
      ]}
    />
  );
}

/** Variables indicadoras de atípicos: contrastes, calibración y cambio de los coeficientes. */
export function OutlierIndicators() {
  const { res, labels } = useApp();
  const oi = res.effects.outlier_indicators;
  if (!oi) return null;
  const L = (t: string) => termLabel(t, labels);
  const models = Object.fromEntries((oi.models as any[]).map((m) => [m.id, m]));
  const top = models.top;
  const cook = models.cook4n;
  const bonf = oi.bonferroni;
  const cal = oi.calibration;
  const changes = top
    ? (top.changes as any[]).map((c) => ({
        ...c,
        coef_cook: cook?.changes.find((x: any) => x.term === c.term),
      }))
    : [];

  return (
    <>
      <Card
        title="Variables indicadoras de atípicos: ¿conviene excluirlos del ajuste?"
        sub="Una indicadora por condado señalado (modelo de cambio en la media). Misma muestra que el modelo final."
      >
        <IndicatorComparison />
        <Explain title="¿Cómo se lee?" open>
          <p>
            La indicadora de un condado vale 1 sólo en él. Su coeficiente δ = y − ŷ<sub>(i)</sub> es el error de predicción del condado cuando no interviene
            en el ajuste, y su t es el residuo estudentizado externamente: <strong>añadir la indicadora equivale a excluir el condado</strong>. El F del bloque
            contrasta si el grupo, en conjunto, se aparta del modelo.
          </p>
          <p>
            Pero los condados se eligieron <em>con los propios residuos</em> (Cook o |r*|): el F nominal no sigue una F y sus p son engañosamente pequeños. El
            p calibrado repite la elección y el contraste sobre {cal ? int(cal.reps) : "—"} muestras simuladas sin atípicos añadidos (bootstrap de los residuos).
          </p>
          <p>
            Se excluiría un grupo sólo si es atípico <em>y</em> su exclusión cambia alguna conclusión (signo o significación de un coeficiente). Con las
            indicadoras de los {int(oi.top)} más influyentes el ajuste mejora (R² aj. {num(top?.r2_adj_final, 3)} → {num(top?.r2_adj, 3)}), pero ningún coeficiente
            cambia más de {num(top?.max_cambio_se, 2)} errores típicos:{" "}
            <strong>{top?.veredicto.vale_la_pena ? "compensa excluirlos" : "no compensa excluirlos"}</strong>; el modelo queda como especificación de sensibilidad. Con los{" "}
            {int(cook?.q)} de Cook &gt; 4/n el R² sube a {num(cook?.r2, 3)} sólo porque cada indicadora reproduce su condado; el F calibrado (p ={" "}
            {pval(cook?.p_calibrado)}) no los distingue de la cola de los residuos y {cook?.bic_mejora ? "el BIC mejora" : "el BIC prefiere el modelo final"}:{" "}
            <strong>{cook?.veredicto.vale_la_pena ? "compensa excluirlos" : "no compensa excluirlos"}</strong>.
          </p>
        </Explain>
      </Card>

      <div className="grid grid-2">
        {top && (
          <Card title={`Indicadoras de los ${int(oi.top)} condados más influyentes`} sub="El p de Bonferroni multiplica por n: se examinan todos los condados.">
            <DataTable
              name="indicadoras-condados"
              rows={top.indicators}
              columns={[
                { key: "county_id", header: "Condado", wrap: true },
                { key: "poblacion", header: "Población", num: true, render: (r: any) => int(r.poblacion) },
                { key: "y", header: "y", num: true, render: (r: any) => num(r.y, 1) },
                { key: "cook", header: "Cook", num: true, render: (r: any) => num(r.cook, 4) },
                { key: "delta", header: "δ = y − ŷ(i)", num: true, render: (r: any) => signed(r.delta, 1) },
                { key: "t", header: "t", num: true, render: (r: any) => num(r.t, 2) },
                { key: "p_bonferroni", header: "p Bonf.", num: true, render: (r: any) => pval(r.p_bonferroni) },
                { key: "atipico", header: "Atípico", render: (r: any) => (r.atipico ? "sí" : "no") },
              ]}
            />
          </Card>
        )}
        <Card title="Contraste de atípicos de Bonferroni" sub={`|r*| > ${num(bonf.critico, 2)}: cuantil 1 − α/(2n) de la t con ${int(bonf.gl)} g.l.`}>
          <DataTable
            name="atipicos-bonferroni"
            rows={bonf.outliers}
            columns={[
              { key: "county_id", header: "Condado", wrap: true },
              { key: "y", header: "y", num: true, render: (r: any) => num(r.y, 1) },
              { key: "stud_resid", header: "r*", num: true, render: (r: any) => num(r.stud_resid, 2) },
              { key: "p_bonferroni", header: "p Bonf.", num: true, render: (r: any) => pval(r.p_bonferroni) },
              { key: "cook", header: "Cook", num: true, render: (r: any) => num(r.cook, 4) },
            ]}
          />
          <p className="small muted">
            Atípico (residuo grande) no es lo mismo que influyente (Cook): el condado con mayor |r*| puede no estar entre los de mayor Cook si su apalancamiento es bajo.
          </p>
        </Card>
      </div>

      {cal && (
        <Card
          title="Calibración del F de las indicadoras elegidas por Cook"
          sub={`Observado: ${int(cal.k_obs)} condados con Cook > 4/n y F = ${num(cal.F_obs, 2)}; los ${int(cal.top)} de mayor Cook, F = ${num(cal.F_top_obs, 2)}.`}
        >
          <DataTable
            name="calibracion-indicadoras"
            csv={false}
            rows={[
              { ley: "Errores normales", ...cal.normal },
              { ley: "Bootstrap de los residuos", ...cal.bootstrap },
            ]}
            columns={[
              { key: "ley", header: "Ley del error simulada" },
              { key: "k_media", header: "Condados con Cook > 4/n", num: true, render: (r: any) => `${num(r.k_media, 0)} ${ci(r.k_lower, r.k_upper, 0)}` },
              { key: "F_media", header: "F de sus indicadoras", num: true, render: (r: any) => `${num(r.F_media, 2)} ${ci(r.F_lower, r.F_upper, 2)}` },
              { key: "p_F", header: "p (Cook > 4/n)", num: true, render: (r: any) => pval(r.p_F) },
              { key: "F_top_media", header: `F de los ${int(cal.top)} mayores`, num: true, render: (r: any) => `${num(r.F_top_media, 2)} ${ci(r.F_top_lower, r.F_top_upper, 2)}` },
              { key: "p_F_top", header: `p (${int(cal.top)} mayores)`, num: true, render: (r: any) => pval(r.p_F_top) },
            ]}
          />
          <Explain>
            <p>
              Aun sin ningún atípico, elegir los residuos mayores y contrastar sus indicadoras da F muy por encima de 1. Frente a errores normales lo
              observado es significativo, lo que sólo confirma que los residuos tienen colas más pesadas que la normal (ya visto en el diagnóstico). Frente a
              su propia distribución (bootstrap), el bloque no es anómalo: son la cola esperable de los errores, no un grupo de condados distinto.
            </p>
          </Explain>
        </Card>
      )}

      {top && (
        <Card title="Coeficientes con y sin indicadoras" sub="Cambio respecto al modelo final, en errores típicos (por conglomerados) del modelo final. † no significativo al 5 %.">
          <DataTable
            name="indicadoras-cambios"
            rows={changes}
            columns={[
              { key: "term", header: "Término", wrap: true, render: (r: any) => L(r.term) },
              { key: "coef_final", header: "Final", num: true, render: (r: any) => num(r.coef_final, 3) + (r.p_final < 0.05 ? "" : " †") },
              { key: "coef", header: `+ ${int(top.q)} indicadoras`, num: true, render: (r: any) => num(r.coef, 3) + (r.p < 0.05 ? "" : " †") },
              { key: "cambio_se", header: "Δ/EE", num: true, render: (r: any) => signed(r.cambio_se, 2) },
              ...(cook
                ? [
                    { key: "cook_coef", header: `+ ${int(cook.q)} indicadoras`, num: true, value: (r: any) => r.coef_cook?.coef, render: (r: any) => num(r.coef_cook?.coef, 3) + (r.coef_cook?.p < 0.05 ? "" : " †") },
                    { key: "cook_cambio", header: "Δ/EE ", num: true, value: (r: any) => r.coef_cook?.cambio_se, render: (r: any) => signed(r.coef_cook?.cambio_se, 2) + (r.coef_cook?.cambia_conclusion ? " *" : "") },
                  ]
                : []),
            ]}
          />
          <p className="small muted">* el coeficiente cambia de signo o de significación al excluir los condados.</p>
        </Card>
      )}
    </>
  );
}
