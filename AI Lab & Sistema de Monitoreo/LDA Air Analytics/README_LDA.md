# Análisis Discriminante Lineal (LDA) y Árboles de Decisión

## Descripción del Proyecto
Este módulo aborda la clasificación del riesgo a la salud pública derivado de la concentración de material particulado fino (PM2.5) en el Área Metropolitana de Monterrey, contrastando un modelo paramétrico rígido frente a un modelo jerárquico no paramétrico. El objetivo consiste en transformar una medición ambiental continua en un diagnóstico binario de peligro para evaluar qué enfoque resulta más efectivo como mecanismo de advertencia temprana.

## Metodología y Escenario Experimental
Para garantizar un entorno de prueba controlado y prevenir la fuga de información:

* **Variable objetivo:** Se construyó una etiqueta binaria tomando como referencia el límite normativo de contaminación para separar los días saludables de los días dañinos. Los identificadores temporales y la variable continua original se descartaron del conjunto de predictores.
* **Partición de datos:** La información se dividió en subconjuntos de entrenamiento y prueba mediante muestreo estratificado con semilla fija, manteniendo la distribución original de clases en ambas particiones para evitar sesgos hacia la clase mayoritaria.
* **Adecuación de supuestos:** Mientras que el árbol de decisión preservó todas las variables debido a su flexibilidad con datos continuos y discretos, en el modelo discriminante se excluyó la variable indicadora de fin de semana para no violar el supuesto de normalidad multivariada continua.

## Modelado y Calibración
Se implementaron dos estrategias con fundamentos geométricos y probabilísticos contrastantes:

* **Análisis Discriminante Lineal (LDA):** Modela la distribución de los predictores bajo el supuesto de normalidad y covarianza compartida, proyectando una frontera lineal de separación entre clases en función de variables meteorológicas y de transporte.
* **Árbol de Decisión:** Segmenta el espacio mediante divisiones binarias recursivas basadas en la reducción de impureza de Gini. Para evitar la memorización de ruido y el sobreajuste del árbol inicial, se aplicó una poda por complejidad de costo identificando el parámetro de penalización que optimiza la estructura final del modelo.

## Evaluación Comparativa y Conclusiones
La evaluación sobre el conjunto de prueba independiente demostró fortalezas complementarias en ambos métodos:

* **Desempeño y geometría:** El árbol de decisión obtuvo una mayor exactitud y especificidad global gracias a su capacidad de generar regiones ortogonales flexibles que capturan interacciones complejas. Por su parte, la rigidez del hiperplano de LDA arrojó una exactitud menor reflejada en su error cuadrático medio.
* **Sensibilidad y salud pública:** A pesar de su menor exactitud general, el modelo LDA superó al árbol en sensibilidad, logrando capturar una mayor proporción de las contingencias reales.
* **Criterio de aplicación:** Para fines de alerta preventiva y protección civil, el LDA resulta más conveniente al minimizar los falsos negativos que dejarían a la población expuesta a la contaminación. En contraste, el árbol de decisión se posiciona como una herramienta óptima para auditoría y visualización de factores de riesgo gracias a su interpretabilidad jerárquica.