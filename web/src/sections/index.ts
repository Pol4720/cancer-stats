import { lazy, type ComponentType, type LazyExoticComponent } from "react";
import type { IconName } from "../components/icons";

export interface SectionDef {
  id: string;
  title: string;
  icon: IconName;
  group: string;
  /** Lo que hay que contar al llegar a la sección en el modo guía. */
  guide: string;
  component: LazyExoticComponent<ComponentType>;
}

export const SECTIONS: SectionDef[] = [
  { id: "inicio", title: "Resumen", icon: "home", group: "Visión general", guide: "La pregunta, el modelo final en cifras (R², R² ajustado, RMSE) y los seis entregables de la orientación, cada uno enlazado a su sección.", component: lazy(() => import("./Overview")) },
  { id: "datos", title: "Datos y procedencia", icon: "database", group: "Datos", guide: "El fichero oficial es el conjunto público de Kaggle: se comprueba celda a celda. La variable añadida Notificadomuerte es casos − muertes y se excluye por fuga.", component: lazy(() => import("./Data")) },
  { id: "depuracion", title: "Validación y depuración", icon: "shield", group: "Datos", guide: "Diecinueve reglas detectan problemas (centinelas, edad en meses, tamaño del hogar dividido por 100…). Once decisiones documentadas los resuelven; tras depurar no queda ningún error.", component: lazy(() => import("./Cleaning")) },
  { id: "ausentes", title: "Datos ausentes", icon: "hole", group: "Datos", guide: "Little no rechaza MCAR para las variables de ausencia aleatoria, pero sí para la incidencia, que falta por estados completos (MAR dado el estado). Casos completos en el modelo e imputación múltiple como sensibilidad.", component: lazy(() => import("./Missing")) },
  { id: "exploracion", title: "Exploración", icon: "chart", group: "Análisis", guide: "Distribución de la mortalidad e inferencia clásica sobre su media, mediana y varianza; relaciones marginales con cada explicativa y diferencias entre regiones.", component: lazy(() => import("./Explore")) },
  { id: "atipicos", title: "Atípicos", icon: "outlier", group: "Análisis", guide: "Detección univariante y multivariante robusta. Los atípicos verificados se conservan; su peso en el modelo se evalúa con medidas de influencia.", component: lazy(() => import("./Outliers")) },
  { id: "modelo", title: "Construcción del modelo", icon: "layers", group: "Modelo", guide: "Poda por FIV, estructura del error según la población, selección hacia atrás con contrastes robustos, confusión de las exposiciones e interacciones con Holm.", component: lazy(() => import("./Building")) },
  { id: "final", title: "Modelo final", icon: "formula", group: "Modelo", guide: "Coeficientes con errores típicos por conglomerados, HC3 y clásicos; salida tipo SPSS e interpretación de cada efecto.", component: lazy(() => import("./Final")) },
  { id: "diagnostico", title: "Diagnóstico", icon: "stethoscope", group: "Modelo", guide: "Linealidad, independencia, homocedasticidad, normalidad, multicolinealidad e influencia: qué se viola y cómo lo absorbe la inferencia.", component: lazy(() => import("./Diagnostics")) },
  { id: "sensibilidad", title: "Sensibilidad", icon: "scale", group: "Modelo", guide: "Las conclusiones se mantienen con MCPF, efectos fijos de estado, modelo mixto, Huber, sin influyentes, con imputación múltiple y con bootstrap.", component: lazy(() => import("./Sensitivity")) },
  { id: "prediccion", title: "Modelo predictivo", icon: "crystal", group: "Modelo", guide: "Ocho modelos comparados con validación cruzada; el elegido se evalúa una vez en prueba, con intervalos conformales.", component: lazy(() => import("./Prediction")) },
  { id: "corridas", title: "Corridas y exportación", icon: "history", group: "Reproducibilidad", guide: "Cada corrida guarda configuración, entorno, huellas y resultados; desde aquí se descargan, incluida la sintaxis SPSS.", component: lazy(() => import("./Runs")) },
  { id: "ejecutar", title: "Configurar y ejecutar", icon: "play", group: "Reproducibilidad", guide: "Cualquier decisión metodológica es una opción de configuración: se cambia, se valida y se vuelve a ejecutar todo el análisis.", component: lazy(() => import("./Configure")) },
];
