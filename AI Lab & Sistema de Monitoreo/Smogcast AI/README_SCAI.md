# Proyecto Final: Benchmark Comparativo y Selección de Modelos

## Descripción del Proyecto
Este proyecto final integra y compara de forma sistemática distintos paradigmas de aprendizaje automático sobre una base de datos unificada de factores climáticos y vehiculares del Área Metropolitana de Monterrey. El objetivo consiste en identificar la arquitectura óptima para clasificar el riesgo ambiental por partículas finas (PM2.5), seleccionando un clasificador capaz de operar como sistema de alerta temprana confiable ante contingencias atmosféricas.

## Metodología y Escenario Experimental
Para asegurar una comparativa justa y reproducible entre todas las arquitecturas evaluadas:

* **Variable objetivo:** Se definió una salida binaria basada en el umbral normativo de contaminación, descartando identificadores temporales y la medición continua original de partículas para evitar fuga de información.
* **Partición y estandarización:** Los datos se dividieron en subconjuntos de entrenamiento y prueba mediante muestreo estratificado con semilla fija, preservando la proporción natural entre días limpios y eventos de riesgo. La normalización de las variables numéricas se ajustó únicamente sobre el conjunto de entrenamiento antes de proyectarse a la prueba ciega.
* **Manejo de supuestos:** Las variables indicadoras categóricas se adaptaron según los requerimientos de cada familia algorítmica, retirándolas en modelos paramétricos continuos para no contravenir el supuesto de normalidad multivariada.

## Modelos Evaluados y Calibración
Se entrenaron y contrastaron cinco familias algorítmicas representativas: Regresión Logística, Análisis Discriminante Lineal (LDA), Árboles de Decisión podados por complejidad de costo, Máquinas de Vectores de Soporte (SVM) y Redes Neuronales Artificiales:

* **Ajuste en entrenamiento:** La selección de hiperparámetros y la búsqueda de los puntos de corte probabilísticos se realizaron exclusivamente dentro del conjunto de entrenamiento mediante validación cruzada estratificada.
* **Prioridad en salud pública:** La optimización se enfocó en maximizar la sensibilidad y el balance armónico de detección, ya que en gestión ambiental el costo social de omitir un día tóxico (falso negativo) es mucho más grave que emitir una falsa alarma preventiva.

## Selección del Modelo Ganador y Evaluación
A partir de la evidencia experimental en validación cruzada, el Análisis Discriminante Lineal (LDA) calibrado con un umbral preventivo resultó ser la alternativa superior:

* **Resistencia al sobreajuste:** Mientras que arquitecturas complejas como las redes neuronales y los vectores de soporte sufrieron memorización de ruido y caídas de desempeño en datos inéditos, el LDA mantuvo estabilidad matemática y consistencia predictiva entre entrenamiento y prueba.
* **Interpretabilidad física:** El hiperplano discriminante lineal modela de forma transparente el impacto de la presión barométrica, la temperatura y las emisiones del transporte sin depender de una estructura de caja negra.
* **Validación independiente:** La evaluación final sobre el conjunto de prueba confirmó que el clasificador seleccionado sostiene una alta tasa de detección de eventos críticos, ofreciendo una herramienta preventiva sustentable para la toma de decisiones ciudadanas y gubernamentales.