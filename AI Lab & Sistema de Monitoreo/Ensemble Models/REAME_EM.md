# Modelos de Ensamble, Máquinas de Vectores de Soporte (SVM) y Redes Neuronales

## Descripción del Proyecto
Este módulo aborda la predicción del riesgo ambiental derivado de concentraciones críticas de partículas finas (PM2.5) en el Área Metropolitana de Monterrey mediante la implementación y calibración de cuatro arquitecturas avanzadas de aprendizaje automático. El objetivo es analizar cómo el incremento en la complejidad estructural y matemática impacta la capacidad de generalización al contrastar modelos de ensamble basados en árboles, clasificadores de margen geométrico y redes neuronales profundas.

## Metodología y Escenario Experimental
Para garantizar un entorno de prueba controlado, reproducible y libre de fuga de datos:

* **Variable objetivo:** Se categorizó la concentración de partículas en una salida binaria según el límite normativo de contaminación para distinguir días saludables de eventos de contingencia, retirando la serie temporal y la medición continua original.
* **Partición y estandarización:** Los datos se dividieron en subconjuntos de entrenamiento y prueba mediante muestreo estratificado con semilla fija, manteniendo la proporción natural de clases en ambas fracciones. La normalización de las variables numéricas se ajustó estrictamente sobre el conjunto de entrenamiento antes de transformar la prueba ciega.
* **Control de sesgo inductivo:** La calibración de hiperparámetros y la búsqueda de los puntos de corte probabilísticos se realizaron dentro del espacio de entrenamiento para evitar el sesgo de optimismo en la evaluación final.

## Construcción y Calibración de Modelos
Se implementaron cuatro familias algorítmicas con mecanismos específicos para mitigar el sobreajuste:

* **Bosques Aleatorios (Random Forest):** Reduce la varianza promediando árboles independientes construidos sobre subconjuntos aleatorios de datos y predictores. Para frenar la memorización de ruido, se limitó la profundidad máxima y se fijó un número mínimo de muestras por hoja.
* **Potenciación del Gradiente (Gradient Boosting):** Construye árboles poco profundos de forma secuencial corrigiendo los errores residuales previos. Se optimizó la tasa de aprendizaje mediante un barrido logarítmico evaluado con el balance armónico de detección y se desplazó el umbral de decisión para priorizar los casos positivos.
* **Máquinas de Vectores de Soporte (SVM):** Utiliza un núcleo de base radial para proyectar los datos a dimensiones superiores y resolver fronteras no lineales. Se calibró el parámetro de costo para balancear la penalización de errores con la tolerancia de un margen suave.
* **Red Neuronal Artificial (MLP):** Arquitectura secuencial multicapa con activaciones lineales rectificadas (ReLU) en las capas intermedias, salida sigmoide probabilística y optimizador adaptativo por momentos. El entrenamiento se reguló mediante parada temprana monitoreando la pérdida de validación para conservar la mejor configuración de pesos antes de sobreajustar.

## Evaluación Comparativa y Conclusiones
El contraste de los modelos sobre datos inéditos arrojó conclusiones clave sobre el balance entre complejidad y rendimiento:

* **Relación entre complejidad y desempeño:** El aumento sustancial en la capacidad de cómputo y número de parámetros no garantizó una superioridad marcada; las arquitecturas más complejas mostraron una tendencia pronunciada a memorizar particularidades del entrenamiento si no se restringían de manera estricta.
* **Compromiso operativo en salud pública:** Ajustar los umbrales de clasificación y la tasa de regularización permitió elevar la sensibilidad en la detección de días dañinos, lo cual resulta prioritario para la protección civil sobre la exactitud superficial del modelo.
* **Interpretabilidad relativa:** Mientras que los ensambles basados en árboles conservan trazabilidad sobre la relevancia de los factores climáticos y vehiculares, los vectores de soporte y las redes neuronales operan como estructuras opacas, por lo que su adopción debe ponderar si la ganancia marginal en predicción justifica la pérdida de interpretabilidad física del fenómeno.