# Clustering Tutorial: K-Means y Clustering Jerárquico

Este proyecto es un cuaderno de Jupyter (`A3.2_Clustering_Tutorial.ipynb`) que enseña, paso a paso, dos formas de **agrupar datos automáticamente** sin decirle a la computadora cuáles son las respuestas correctas. A esto se le llama **aprendizaje no supervisado**.

## ¿De qué trata, en pocas palabras?

Imagina que tienes una caja con muchas fotos mezcladas y quieres ordenarlas en montones de fotos parecidas, pero nadie te dijo cuántos montones hay ni qué tema tiene cada uno. Los algoritmos de *clustering* hacen eso con datos: encuentran grupos de elementos parecidos entre sí.

El cuaderno usa dos métodos:

| Método | Idea sencilla |
|---|---|
| **K-Means** | Tú eliges cuántos grupos quieres (*k*). El algoritmo coloca un "centro" por grupo y asigna cada punto al centro más cercano. Luego mueve los centros al promedio de su grupo y repite hasta que nada cambia. |
| **Clustering jerárquico** | Empieza con cada elemento solo y va uniendo los más parecidos, de dos en dos, hasta formar un solo grupo grande. El resultado es un árbol llamado **dendrograma**, que se puede "cortar" a distintas alturas para obtener más o menos grupos. |

## Los datos que se usan

1. **Datos sintéticos:** 1,000 puntos inventados en 2 dimensiones, con 3 grupos. Sirven para ver cómo funciona K-Means en un caso fácil de dibujar.
2. **NCI60:** datos reales de la librería `ISLP`. Son 64 líneas celulares de cáncer, cada una descrita por la actividad de 6,830 genes. Cada muestra tiene una etiqueta con su tipo de cáncer (mama, colon, leucemia, melanoma, etc.). **Esas etiquetas no se usan para agrupar**; solo se usan al final para comprobar si los grupos encontrados se parecen a los tipos reales.

## Qué se hace, parte por parte

**Parte I: K-Means con datos sintéticos**
- Se crean y dibujan los puntos.
- Se programa K-Means a mano para ver cómo se mueven los centros hasta estabilizarse (en este caso, 4 iteraciones).
- Se repite con `scikit-learn`, que prueba varios inicios aleatorios y se queda con el mejor.

**Parte II: K-Means con NCI60**
- Se estandarizan los datos para que todos los genes tengan una escala comparable.
- Se prueba con 3, 4 y 5 grupos y se comparan dos medidas:
  - **Inercia:** qué tan juntos están los puntos de su centro (menor es mejor, pero siempre baja al subir *k*).
  - **Silhouette:** qué tan bien separados están los grupos (cerca de 1 es muy bueno, cerca de 0 es que se traslapan).
- Como hay 6,830 genes, se usa **PCA** para resumir la información en menos dimensiones (32 componentes explican al menos el 80 % de la variación) y poder graficarla.
- Se compara cada grupo con los tipos de cáncer reales y se dibuja un **heatmap** con los 50 genes que más distinguen a dos grupos.

**Parte III: Clustering jerárquico con NCI60**
- Se construyen dendrogramas y se cortan para obtener 2, 3 y 4 grupos.
- Se comparan distintas formas de medir parecido (distancia **euclidiana** contra **correlación**) y distintas reglas para unir grupos (**complete** contra **average**).
- Se crean 3 grupos y se comparan con los tipos de cáncer reales.

## Resultados principales

- En los datos sintéticos, K-Means separa muy bien los 3 grupos (silhouette de 0.765).
- En NCI60 los silhouette son bajos (alrededor de 0.08 con los datos originales y 0.13 con PCA). Los grupos no están fuertemente separados, así que elegir el número "correcto" de grupos no es claro.
- Con las métricas probadas, el mejor *k* fue 4 con los datos estandarizados y 5 después de PCA. Es una referencia, no una verdad.
- Algunos tipos de cáncer se agrupan claramente. Por ejemplo, el cáncer de colon y el de riñón (renal) caen casi por completo en un mismo grupo, y el melanoma se concentra en otro. Otros, como mama y pulmón (NSCLC), se reparten entre varios grupos.
- En el árbol jerárquico, las leucemias quedan juntas en un grupo propio.
- Cambiar la métrica o la regla de unión cambia la forma del dendrograma, así que el resultado depende de cómo se defina "parecido".

## Cómo ejecutarlo

1. Abre `A3.2_Clustering_Tutorial.ipynb` en Jupyter o Google Colab.
2. Ejecuta las celdas de arriba hacia abajo. La primera instala `ISLP`.
3. Librerías usadas: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy` e `ISLP`.

## Referencias

- Martínez Torteya, A. *C3.3 K-Means Clustering* y *C3.4 Hierarchical Clustering* (presentaciones de clase, Universidad de Monterrey).
- James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*. Springer.
