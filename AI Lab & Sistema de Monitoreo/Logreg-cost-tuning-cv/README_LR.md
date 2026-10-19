# Regresión Logística y Validación Cruzada

## Descripción del Proyecto
Este módulo establece la línea base predictiva mediante un modelo de regresión logística para clasificar el riesgo ambiental por partículas finas (PM2.5) en el Área Metropolitana de Monterrey. El objetivo consiste en determinar si la cuenca urbana superará el umbral crítico de contaminación diaria, transformando el problema en una clasificación binaria entre días saludables y días dañinos para la salud pública.

## Diseño Experimental y Partición de Datos
Para garantizar la fiabilidad del estudio y prevenir la fuga de información:

* **Variable objetivo:** Se construyó una variable binaria a partir del límite normativo de contaminación. La variable continua original de material particulado se excluyó de los predictores para evitar filtraciones de la respuesta.
* **Partición estratificada:** Los datos se dividieron en subconjuntos de entrenamiento y prueba mediante muestreo estratificado y semilla fija, preservando la proporción natural entre días limpios y eventos de riesgo para evitar sesgos predictivos.
* **Estandarización:** Los predictores continuos se escalaron ajustando media y desviación estándar únicamente sobre los datos de entrenamiento, proyectando posteriormente dicha transformación al conjunto de prueba.

## Validación Cruzada
La estabilidad del modelo se evaluó exclusivamente dentro del conjunto de entrenamiento mediante validación cruzada estratificada de cinco pliegues:

* Esta división rotativa permite estimar el desempeño esperado frente a datos inéditos sin contaminar la prueba ciega.
* La calibración del algoritmo prioriza la detección de la clase minoritaria de riesgo ambiental frente a la clase mayoritaria de días limpios, asegurando un compromiso balanceado entre sesgo y varianza antes de cualquier decisión final.