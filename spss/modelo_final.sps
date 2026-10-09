* ==========================================================================.
* Reproducción en SPSS del modelo final MCO (proyecto cancer-stats).
* Generado automáticamente a partir de la corrida 20261009-034617-f8341612.
* Los coeficientes y errores típicos clásicos deben coincidir con la tabla
* «salida tipo SPSS» del informe. Ejecutar con el fichero oficial en la
* misma carpeta que esta sintaxis.
* ==========================================================================.

SET DECIMAL=DOT.
GET FILE='../data/raw/practica.sav'.
DATASET NAME practica WINDOW=FRONT.

* D01: Geography separada en condado y estado (por la última coma).
STRING state (A40).
COMPUTE #coma = CHAR.RINDEX(Geography, ',').
COMPUTE state = LTRIM(CHAR.SUBSTR(Geography, #coma + 1)).
EXECUTE.

* D02: Notificadomuerte = avgAnnCount - avgDeathsPerYear (fuga): no entra en el modelo.

* D03: valor centinela de incidenceRate.
IF (ABS(incidenceRate - 453.5494000) < 0.000001) incidenceRate = $SYSMIS.

* D04: edad mediana registrada en meses.
IF (MedianAge > 100) MedianAge = RND(MedianAge / 12 * 10) / 10.
* D05: tamaño medio del hogar dividido por 100.
IF (AvgHouseholdSize < 1) AvgHouseholdSize = AvgHouseholdSize * 100.
* D06: PctSomeCol18_24 recuperada por identidad contable.
IF (MISSING(PctSomeCol18_24)) PctSomeCol18_24 = MAX(0, 100 - PctNoHS18_24 - PctHS18_24 - PctBachDeg18_24).
* D07: población nativa o multirracial (residuo racial).
COMPUTE PctNativeMulti = MAX(0, 100 - (PctWhite + PctBlack + PctAsian + PctOtherRace)).
* D08: incidencia inverosímil (mortalidad mayor que incidencia).
IF (TARGET_deathRate / incidenceRate > 1) incidenceRate = $SYSMIS.
* D09: región censal.
STRING region (A12).
IF (ANY(RTRIM(state), 'Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island', 'Vermont', 'New Jersey', 'New York', 'Pennsylvania')) region = 'Noreste'.
IF (ANY(RTRIM(state), 'Illinois', 'Indiana', 'Michigan', 'Ohio', 'Wisconsin', 'Iowa', 'Kansas', 'Minnesota', 'Missouri', 'Nebraska', 'North Dakota', 'South Dakota')) region = 'Medio Oeste'.
IF (ANY(RTRIM(state), 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Maryland', 'North Carolina', 'South Carolina', 'Virginia', 'West Virginia', 'Alabama', 'Kentucky', 'Mississippi', 'Tennessee', 'Arkansas', 'Louisiana', 'Oklahoma', 'Texas')) region = 'Sur'.
IF (ANY(RTRIM(state), 'Arizona', 'Colorado', 'Idaho', 'Montana', 'Nevada', 'New Mexico', 'Utah', 'Wyoming', 'Alaska', 'California', 'Hawaii', 'Oregon', 'Washington')) region = 'Oeste'.
* D10: transformaciones.
COMPUTE logPop = LN(popEst2015).
COMPUTE logIncome = LN(medIncome).
COMPUTE logStudy = LN(1 + studyPerCap).
EXECUTE.

* Indicadoras de región (referencia: Sur).
COMPUTE reg_NE = (RTRIM(region) = 'Noreste').
COMPUTE reg_MO = (RTRIM(region) = 'Medio Oeste').
COMPUTE reg_OE = (RTRIM(region) = 'Oeste').

* Explicativas continuas centradas en la media de la muestra del modelo final.
COMPUTE c_incidenceRate = incidenceRate - 447.9725000000.
COMPUTE c_MedianAge = MedianAge - 40.7675700000.
COMPUTE c_AvgHouseholdSize = AvgHouseholdSize - 2.5373700000.
COMPUTE c_PctHS18_24 = PctHS18_24 - 35.2442300000.
COMPUTE c_PctBachDeg25_Over = PctBachDeg25_Over - 13.1516500000.
COMPUTE c_PctUnemployed16_Over = PctUnemployed16_Over - 8.0428870000.
COMPUTE c_PctPublicCoverageAlone = PctPublicCoverageAlone - 19.5347900000.
COMPUTE c_PctOtherRace = PctOtherRace - 1.9884140000.
COMPUTE c_BirthRate = BirthRate - 5.5805040000.
COMPUTE c_PctHS25_Over = PctHS25_Over - 34.9255300000.

* Interacciones (productos de variables centradas).
COMPUTE c_PctBachDeg25_Over_x_NE = c_PctBachDeg25_Over * reg_NE.
COMPUTE c_PctBachDeg25_Over_x_MO = c_PctBachDeg25_Over * reg_MO.
COMPUTE c_PctBachDeg25_Over_x_OE = c_PctBachDeg25_Over * reg_OE.
COMPUTE c_PctUnemployed16_Over_x_NE = c_PctUnemployed16_Over * reg_NE.
COMPUTE c_PctUnemployed16_Over_x_MO = c_PctUnemployed16_Over * reg_MO.
COMPUTE c_PctUnemployed16_Over_x_OE = c_PctUnemployed16_Over * reg_OE.
EXECUTE.

* Modelo final por mínimos cuadrados ordinarios, con los diagnósticos de la
* orientación: linealidad, independencia (Durbin-Watson), homocedasticidad,
* normalidad de los residuos, colinealidad (tolerancia, FIV e índices de condición)
* y casos atípicos e influyentes.
REGRESSION
  /MISSING LISTWISE
  /STATISTICS COEFF OUTS CI(95) R ANOVA COLLIN TOL
  /CRITERIA=PIN(.05) POUT(.10)
  /NOORIGIN
  /DEPENDENT TARGET_deathRate
  /METHOD=ENTER
    c_incidenceRate c_MedianAge c_AvgHouseholdSize c_PctHS18_24
    c_PctBachDeg25_Over c_PctUnemployed16_Over c_PctPublicCoverageAlone
    c_PctOtherRace c_BirthRate c_PctHS25_Over reg_NE reg_MO reg_OE
    c_PctBachDeg25_Over_x_NE c_PctBachDeg25_Over_x_MO
    c_PctBachDeg25_Over_x_OE c_PctUnemployed16_Over_x_NE
    c_PctUnemployed16_Over_x_MO c_PctUnemployed16_Over_x_OE
  /SCATTERPLOT=(*ZRESID, *ZPRED) (*SRESID, *ZPRED)
  /RESIDUALS DURBIN HISTOGRAM(ZRESID) NORMPROB(ZRESID)
  /CASEWISE PLOT(ZRESID) OUTLIERS(3)
  /SAVE PRED ZRESID SRESID COOK LEVER SDBETA.

* Contraste de Breusch-Pagan (versión de Koenker) a mano: regresión de los
* residuos al cuadrado sobre las explicativas; LM = n·R².
COMPUTE res2 = (ZRE_1) ** 2.
REGRESSION /DEPENDENT res2 /METHOD=ENTER
    c_incidenceRate c_MedianAge c_AvgHouseholdSize c_PctHS18_24
    c_PctBachDeg25_Over c_PctUnemployed16_Over c_PctPublicCoverageAlone
    c_PctOtherRace c_BirthRate c_PctHS25_Over reg_NE reg_MO reg_OE
    c_PctBachDeg25_Over_x_NE c_PctBachDeg25_Over_x_MO
    c_PctBachDeg25_Over_x_OE c_PctUnemployed16_Over_x_NE
    c_PctUnemployed16_Over_x_MO c_PctUnemployed16_Over_x_OE.

