# GASTROCLUST: Gastric Cancer Clustering Analysis

**Autor:** Bruno Morales Rosales

Este proyecto busca **grupos de tumores de estómago con perfiles parecidos**, usando solo la actividad de sus genes. Es un ejercicio de **aprendizaje no supervisado**: la computadora agrupa los tumores sin saber nada de la enfermedad de cada paciente. Después se revisa si los grupos que encontró tienen sentido biológico y clínico.

## ¿De qué trata, en pocas palabras?

El cáncer de estómago no es una sola enfermedad: unos tumores crecen formando glándulas (tipo *intestinal*), otros se esparcen en células sueltas (tipo *difuso*), y otros son distintos por dentro aunque se vean parecidos. Medir qué genes están "encendidos" en cada tumor (con una técnica llamada RNA-seq) permite compararlos de forma más fina que con el microscopio.

La idea del proyecto es como ordenar una caja de fotos mezcladas sin que nadie te diga las categorías: se agrupan las que se parecen y luego se averigua qué tienen en común.

## Los datos

Vienen del proyecto **TCGA** (Stomach Adenocarcinoma, STAD), descargados desde la plataforma UCSC Xena. Están en la carpeta `Datos/`:

| Archivo | Qué contiene |
|---|---|
| `TCGA-STAD_star_tpm.tsv.xz` | Expresión de 60,660 genes en 448 muestras (412 tumores y 36 tejidos normales). Está comprimido (ver nota abajo). |
| `TCGA-STAD.clinical.tsv` | Datos clínicos de los pacientes: edad, sexo, estadio, tipo de tumor, etc. |
| `TCGA-STAD.survival.tsv` | Cuánto tiempo siguió vivo cada paciente y si falleció o no. |
| `gencode.v36.annotation.gtf.gene.probemap` | Tabla para traducir los códigos de los genes (`ENSG…`) a sus nombres (por ejemplo, *CDH1*). |

> **Nota sobre el archivo de expresión:** el original pesa 368 MB y GitHub no admite archivos de más de 100 MB. Se comprimió con `xz` (58 MB) sin perder información. El notebook lo encuentra y lo lee tal cual, sin descomprimirlo.

## Qué se hace, paso a paso

1. **Explorar los datos:** cuántas muestras hay, cuántas son tumores, qué variables clínicas sirven y si hay datos faltantes o muestras raras.
2. **Preparar los datos:** se usan solo los 412 tumores primarios, se quitan los genes casi apagados o poco confiables, se corrige el efecto del ARN mitocondrial y se escogen los 5,000 genes que más varían entre tumores.
3. **PCA (reducción de dimensiones):** resume miles de genes en unos pocos "ejes" que se pueden graficar y entender. Se usan 10 componentes.
4. **Clustering:** se comparan dos métodos, **K-means** (se elige cuántos grupos y el algoritmo coloca los centros) y **jerárquico con enlace de Ward** (va uniendo los tumores más parecidos y forma un árbol). Se elige K-means con 4 grupos.
5. **Interpretar los grupos con genes:** se buscan los genes que más distinguen a cada grupo para saber qué tipo de tumor representa.
6. **Interpretar con la clínica:** se revisa si los grupos se relacionan con el tipo de tumor, el estadio, la edad o la supervivencia.
7. **Comparar con la literatura:** se contrasta con el artículo de referencia de TCGA (Nature, 2014).

## Resultados principales

- **Los tumores forman un continuo, no islas separadas.** Los grupos se traslapan (silueta media de 0.16), así que las fronteras entre ellos son difusas.
- Los tres ejes que más diferencian a los tumores son: **cuánto tejido de soporte (estroma) frente a células tumorales en división**, **cuánta infiltración de células inmunes** y **qué tan diferenciado está el epitelio**.
- Los 4 grupos encontrados con K-means son estables y cada uno tiene un perfil reconocible:

| Grupo | Tumores | Perfil |
|---|---|---|
| **C1** | 104 | Epitelial de tipo intestinal |
| **C2** | 108 | Proliferativo y con pocas células inmunes ("inmune-frío") |
| **C3** | 126 | Muy inflamado, con muchas células inmunes |
| **C4** | 74 | Estromal / mesenquimal |

- **Los grupos se relacionan con la clínica, aunque el algoritmo no vio esos datos.** C4 concentra los tumores de tipo difuso y los pacientes son unos 6 años más jóvenes. C1 tiene más tumores intestinales y tempranos. También hay relación con la profundidad de invasión, el estadio y la región del estómago.
- **La supervivencia no difiere de forma significativa entre grupos** (p = 0.14). El orden observado (C1 mejor, C4 y C2 peor) coincide con lo esperado, pero no se puede afirmar con estos datos.
- **K-means y el jerárquico coinciden en los extremos pero no en el centro** (acuerdo moderado, ARI = 0.29), por eso conviene no presentar un solo resultado como "los subtipos" definitivos.
- C4 se parece al subtipo *genomically stable* descrito por TCGA, pero no hay una correspondencia uno a uno, porque esos subtipos se definieron con datos de ADN.

## Limitaciones

- La estructura de los grupos es débil y depende del método y del número de genes usados.
- El RNA-seq de tejido mezcla varios tipos de células (tumorales, inmunes, estroma), y no se puede separar cuánto es del tumor y cuánto del tejido de alrededor.
- El **centro de origen** de la muestra también se asocia con los grupos y no se corrigió, así que parte de la señal puede ser técnica.
- Solo se usó un tipo de dato (expresión); faltan mutaciones, metilación y estado de los virus/inestabilidad que definen los subtipos oficiales.
- No hay una cohorte externa para validar los resultados y el seguimiento de los pacientes es corto.

## Cómo ejecutarlo

1. Abre `TCGA_STAD_clustering_no_supervisado.ipynb` en Jupyter o Google Colab.
2. Deja la carpeta `Datos/` junto al notebook (el notebook la busca en `./Datos`). En Colab, sube la carpeta o cambia `DATA_DIR`.
3. Ejecuta las celdas en orden.
4. Librerías: `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`.

## Archivos del proyecto

```
GASTROCLUST – Gastric Cancer Clustering Analysis/
├── README.md
├── TCGA_STAD_clustering_no_supervisado.ipynb
└── Datos/
    ├── TCGA-STAD_star_tpm.tsv.xz
    ├── TCGA-STAD.clinical.tsv
    ├── TCGA-STAD.survival.tsv
    └── gencode.v36.annotation.gtf.gene.probemap
```

## Referencias principales

- The Cancer Genome Atlas Research Network (2014). *Comprehensive molecular characterization of gastric adenocarcinoma*. Nature, 513, 202–209. https://doi.org/10.1038/nature13480
- Cristescu, R. et al. (2015). *Molecular analysis of gastric cancer identifies subtypes associated with distinct clinical outcomes*. Nature Medicine, 21, 449–456. https://doi.org/10.1038/nm.3850
- Goldman, M. J. et al. (2020). *Visualizing and interpreting cancer genomics data via the Xena platform*. Nature Biotechnology, 38, 675–678. https://doi.org/10.1038/s41587-020-0546-8
