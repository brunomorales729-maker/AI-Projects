# Auditoría técnica y registro de reestructuración

Fecha de auditoría: 4 de noviembre de 2026  
Origen: `Python-Chess-Game.zip` (estructura original conservada sin modificar; la entrega es un proyecto nuevo).

## 1. Hallazgos del proyecto original

| Hallazgo | Evidencia | Acción |
|---|---|---|
| Dos generadores idénticos | `training/dataset.py` y `training/generate_dataset.py` tenían el mismo contenido byte por byte (308 líneas) | Se conserva una sola implementación de generación; `dataset.py` ahora es el lector/procesador real. |
| Entrenamiento roto | `train_supervised.py` importaba `training.dataset_v2`, que no está incluido | Se reemplazó por `training/train.py` y `training/dataset.py`. |
| Checkpoints incompatibles | Tres checkpoints `chess_policy*.pth`: 13 canales, salida de 4,096 acciones, sin Value Head; un checkpoint ResNet: 18 canales, 20,480 acciones y Policy + Value | Se descartó la familia antigua de política-only y se conservó como inicialización la ResNet compatible. |
| Datasets redundantes | Había 6 JSONL Stockfish y un JSONL self-play en rutas distintas | Se consolidaron tres fuentes Stockfish seleccionadas en un solo JSONL; self-play e históricos quedan fuera de la estructura final. |
| V4 sesgado por turno | Las 10,000 posiciones de V4 tenían `turn=white`; el muestreo `ply % 2 == 0` seleccionaba solo una paridad | El nuevo generador muestrea cada ply, incluyendo blancas y negras. La limitación del dataset ya generado se documenta más abajo. |
| Transposiciones y reloj de 50 movimientos | V4 presentaba 563 repeticiones por tablero/turno/derechos de enroque/en passant al ignorar los contadores | El estado definitivo se deduplica por los primeros cinco campos FEN (incluye medio movimiento); el agrupamiento por partidas relaciona los primeros cuatro para evitar fuga por transposición. Se preservan 588 variantes de reloj relevantes. |
| Evaluaciones MultiPV desordenadas | 1,885 de 10,000 registros V4 no estaban en orden descendente por score | Los candidatos consolidados se ordenaron por `score_cp` descendente; el validador exige ese orden. |
| Validador desalineado | El validador original apuntaba a `stockfish_dataset.jsonl`, mientras el generador escribía `stockfish_dataset_v4.jsonl` | Un solo nombre de dataset, compartido desde `config.py`. |
| Evaluación vacía | `training/evaluate.py` tenía 0 bytes | Se implementó evaluación contra niveles Stockfish, CPL, coincidencia de mejor jugada y partidas truncadas separadas. |
| GUI limitada | La GUI AlphaZero original fijaba humano=blancas, IA=negras, 100 simulaciones y promoción automática a dama | La GUI final permite escoger color, reiniciar, ver historial y movimientos legales, seleccionar promoción por teclado y configurar PUCT. |
| Recursos duplicados | Había dos carpetas de SVG de piezas (`pieces/` y `chess_pieces/`) | Se conserva una sola copia en `assets/pieces/`. |
| Dependencias ausentes | El ZIP no incluía `requirements.txt`; Stockfish tampoco estaba incluido | Se añadió `requirements.txt`; Stockfish se configura mediante ruta/variable de entorno. |

### Checkpoints originales

- `models/chess_policy.pth`, `chess_policy_v2.pth`, `chess_policy_v3.pth`: los tres comparten la arquitectura antigua de 13 canales y 4,096 acciones (4,221,184 parámetros). No representan el mismo contrato de entrada/salida que el sistema AlphaZero y no codifican explícitamente cuatro tipos de promoción ni valor de posición.
- `models/supervised/chess_resnet_best.pth`: checkpoint compatible con la arquitectura conservada. Su metadata indicaba época 21, `loss=6.3271`, `policy_loss=6.2588`, `value_loss=0.0683`, `accuracy=0.09`, `value_mae=0.1826`. Se verificó que su `state_dict` carga sin claves faltantes en la ResNet final. Fue utilizado como inicialización del ajuste supervisado.

### Datasets originales

- `stockfish_dataset_v4.jsonl`: 10,000 posiciones, 20/60/20 por etapa, todos los FEN con blancas al turno; 563 posiciones se repetían por transposición; 18.85% de registros con candidatos fuera de orden.
- `stockfish_dataset.jsonl`: 500 posiciones, contiene ambos turnos y evaluaciones con mayor profundidad mediana que V4.
- `stockfish_dataset_previous.jsonl`: 500 posiciones Stockfish adicionales, con etapas y candidatos estructurados.
- `stockfish_dataset_v2.jsonl`: 100 posiciones, carecía de finales y tenía errores de ordenación MultiPV; excluido.
- `stockfish_dataset_backup.jsonl`: 100 posiciones, no tenía campo de etapa y presentaba errores de ordenación; excluido.
- `stockfish_dataset_v4_backup_*.jsonl`: copia parcial/histórica de V4; excluida para no duplicar su contenido.
- `data/processed/self_play.jsonl`: 3,040 registros de autojuego, no son etiquetas Stockfish; excluido por no corresponder al flujo supervisado solicitado.

## 2. Arquitectura final

```text
ChessAI/
├── agent/
│   ├── encoding.py
│   ├── network.py
│   ├── policy.py
│   └── mcts.py
├── training/
│   ├── generate_dataset.py
│   ├── dataset.py
│   ├── validate_dataset.py
│   ├── train.py
│   └── evaluate.py
├── data/stockfish_dataset.jsonl
├── models/chess_model_best.pth
├── gui/main.py
├── assets/pieces/*.svg
├── tests/test_core.py
├── config.py
├── requirements.txt
├── README.md
└── AUDIT_REPORT.md
```

La inferencia importa `ChessPolicy`/`MCTS` y no contiene llamadas al motor UCI. Stockfish solo aparece en generación y evaluación. PUCT es opcional: con cero simulaciones se usa la red directamente; con simulaciones positivas se combina prior de política, valor neuronal y exploración PUCT.

## 3. Dataset consolidado

Se unieron las tres fuentes Stockfish seleccionadas y se eliminaron posiciones repetidas por los primeros cinco campos FEN (incluye el reloj de medio movimiento). Cuando una misma posición aparecía en más de una fuente, se conservó el registro con mayor profundidad mediana, después mayor cantidad de candidatos y, como desempate, profundidad máxima. Los candidatos se ordenaron por evaluación descendente desde la perspectiva del jugador al turno.

| Métrica | Resultado |
|---|---:|
| Registros finales | **10,996** |
| Opening | 2,296 (20.9%) |
| Middlegame | 6,400 (58.2%) |
| Endgame | 2,300 (20.9%) |
| Turno blanco en FEN | 10,746 (97.7%) |
| Turno negro en FEN | 250 (2.3%) |
| Posiciones duplicadas por clave | 0 |
| Evaluaciones candidatas de Stockfish | **51,357** |
| Grupos de partida reconstruidos | **241** |
| Registros con `game_id` y `ply` | 10,996 / 10,996 |
| Registros mal formados | 0 |
| Candidatos duplicados en la misma posición | 0 |
| Errores de orden de candidatos | 0 |
| Rango de profundidad reportada | 7–245; mediana 11 |
| Rango de score (POV del jugador al turno) | -9,999 a +9,999 cp |

La red usa orientación canónica relativa al jugador al turno, así que las posiciones se representan con canales `propios/oponente` y no dependen de que mueva blanco o negro. Eso reduce el impacto del desequilibrio de turnos en el tensor. No obstante, la proporción de FEN blancos sigue siendo una limitación de cobertura del conjunto incluido.

#### Reconstrucción de grupos de partida y split sin fuga

Los tres JSONL originales conservaban el orden de escritura por partida, aunque no guardaban `game_id`. Se recuperaron las identidades asociando los 10,996 registros consolidados con los originales por clave de posición y firma exacta de candidatos; se detectaron límites de partida por el reinicio descendente del contador `fullmove`. Después se unieron mediante componentes conectados las partidas que compartían tablero, turno, derechos de enroque y casilla en passant (primeros cuatro campos FEN), para que una posición transpuesta con distinto reloj no separase grupos relacionados. El resultado son **241 grupos de partida** a partir de 245 secuencias originales. Si una partida muy corta no provoca reinicio del contador, el método puede fusionar dos grupos adyacentes, pero no dividir posiciones de una misma partida; es una decisión conservadora contra fugas.

El dataset final deduplica por los primeros cinco campos FEN: conserva el reloj de medio movimiento porque interviene en la regla de 50 movimientos y en el canal 18 del tensor, e ignora solo el contador de jugada completa. Así retiene 588 variantes de reloj que una deduplicación de solo cuatro campos habría colapsado. Guarda `game_id` y `ply`. El split reservado por partida completa contiene **9,258 registros de entrenamiento (208 grupos)** y **1,738 de validación (33 grupos)**. Se verifica que no haya intersección ni de `game_id` ni de claves de posición. Para datasets nuevos, el generador emite `game_id` y `ply` desde el momento de creación.

### Targets del aprendizaje

- Policy: softmax de las evaluaciones Stockfish de los candidatos, con temperatura de 100 cp; permite que varios candidatos fuertes contribuyan a la etiqueta.
- Value: `tanh(best_score_cp / 600)`, usando el mejor score desde el lado al turno.
- Los índices de acciones son `origen × destino × promoción`, con cinco opciones (sin promoción, dama, torre, alfil, caballo), por lo que hay 20,480 logits.
- La carga de datos conserva estados y listas dispersas de candidatos; no crea una matriz densa de política ni máscaras legales de 20,480 entradas para todo el dataset. En inferencia, la política sí enmascara todos los movimientos ilegales.

## 4. Modelo final

- 18 planos de entrada: 12 canales de piezas relativas, 4 de derechos de enroque relativos, 1 de en passant y 1 del contador de medio movimiento.
- Stem convolucional + 4 bloques residuales de 64 canales.
- Policy Head bilineal con embeddings de origen/destino y cinco opciones de promoción.
- Value Head convolucional + MLP + `tanh`.
- Parámetros entrenables: **324,233**.
- CUDA es seleccionado automáticamente cuando PyTorch lo detecta. El entrenamiento activa AMP en CUDA y mantiene precisión completa en CPU.

No se aumentó el tamaño de la red: se conservó la ResNet Policy + Value existente, que es la única arquitectura/checkpoint original compatible con la codificación final.

## 5. Entrenamiento realizado y estado

Se volvió a ajustar desde el checkpoint ResNet + Policy/Value original utilizando el dataset definitivo de 10,996 estados y el split por grupos de partida. El entorno de auditoría solo dispone de PyTorch **CPU-only**, no tiene CUDA y cada ejecución tiene un límite. El mejor checkpoint guardado corresponde a la **época 4**; la época 5 no mejoró y una continuación posterior hasta la época 8 no produjo un checkpoint superior antes de alcanzar el límite de ejecución. No se activó early stopping (paciencia configurada: 7).

| Métrica | Entrenamiento | Validación |
|---|---:|---:|
| Total loss | 4.0724 | **5.3009** |
| Policy loss | 4.0161 | 5.1702 |
| Value loss | 0.0563 | 0.1307 |
| Policy top-1 | 18.00% | 5.58% |
| Value MAE | 0.1704 | 0.2592 |

El checkpoint guarda el hash SHA-256 del dataset, optimizer, scheduler y estado de GradScaler. Se corrigió también la reanudación CPU→CUDA: un scaler vacío de un checkpoint CPU ya no se carga sobre un GradScaler habilitado. `--resume` rechaza cualquier dataset distinto y conserva el estado del mejor checkpoint. La validación se reserva por partida completa, no por renglón.

**Interpretación:** es un ajuste supervisado inicial y comprobable, no una convergencia final ni una prueba de fuerza. Las épocas 6–8 sugieren sobreajuste incipiente; el scheduler/early stopping están preparados para continuar en la RTX 4050, comenzando con batch 32 y reduciendo a 16/8 si falta VRAM. No se afirma que el modelo esté optimizado.

### Comparación directa con el checkpoint original

El checkpoint inicial compatible se reevaluó desde cero en **el mismo split final reservado por partidas de 1,738 posiciones** que el modelo ajustado:

| Métrica | ResNet original | Modelo ajustado, época 4 |
|---|---:|---:|
| Total loss | 8.0380 | **5.3009** |
| Policy loss | 7.8635 | 5.1702 |
| Value loss | 0.1745 | 0.1307 |
| Policy top-1 | 2.70% | **5.58%** |
| Value MAE | 0.3027 | **0.2592** |

La pérdida total bajó **34.1%**, la exactitud top-1 aumentó **2.06×** y el MAE de valor bajó **14.4%**. Esta comparación usa partidas completas no vistas y sustituye todas las métricas preliminares de splits anteriores y del dataset previo a la corrección del reloj de medio movimiento. Demuestra mejor ajuste a las etiquetas de Stockfish en estas posiciones reservadas; **no** permite inferir Elo, W/D/L ni fuerza competitiva. El checkpoint inicial reporta época 21 y `loss=6.3271` en su metadata, pero no contiene hash del dataset ni identificación del split histórico; ese valor no se usa como baseline directo.

## 6. Pruebas y limitaciones de esta auditoría

| Prueba | Resultado |
|---|---|
| Compilación sintáctica de Python | Correcta (`compileall`) |
| Tests unitarios | **7/7 correctos** |
| Codificación de tablero inicial | Correcta, forma (18, 8, 8) |
| Canonicalización de `e2e4` blanco / `e7e5` negro | Mismo índice de acción relativo |
| Codificación de promociones | Q/R/B/N generan acciones distintas |
| Forward de red | Policy `(2, 20480)`, Value `(2,)` |
| Presupuesto del modelo | Menos de 500,000 parámetros |
| Compatibilidad del checkpoint original ResNet | Todas las claves coinciden |
| Dataset: estructura, duplicados, turnos y orden MultiPV | Correcto según validación estructural |
| FEN legales y movimientos candidatos legales | **No ejecutado**: faltaba `python-chess` |
| Interfaz gráfica | **No ejecutada**: faltaba `pygame-ce` |
| Ejecución contra Stockfish | **No ejecutada**: el binario no venía en el ZIP |
| CUDA / AMP en RTX 4050 | **No ejecutado**: PyTorch del entorno es CPU-only |
| Torneo para medir Elo/W-D-L/CPL real | Pendiente de ejecutar en el equipo con Stockfish |

La instalación de dependencias fue intentada, pero PyPI no era accesible desde este entorno. Al ejecutar localmente `pip install -r requirements.txt`, el validador hará las verificaciones completas; después se debe correr `training.evaluate` para medir resultados reales contra varios niveles Stockfish.

## 7. Archivos retirados, fusionados y conservados

La entrega nueva no contiene las versiones viejas; el ZIP original subido permanece intacto como respaldo externo del trabajo.

| Original | Disposición |
|---|---|
| `agent/alpha_zero/encoding.py` | Fusionado y adaptado como `agent/encoding.py` |
| `agent/alpha_zero/network.py` | Fusionado y adaptado como `agent/network.py` |
| `agent/alpha_zero/policy.py` | Reemplazado por `agent/policy.py` |
| `agent/alpha_zero/mcts.py` | Conservado y endurecido como `agent/mcts.py` |
| `training/generate_dataset.py` | Único generador final |
| `training/dataset.py` (duplicado antiguo) | Reemplazado por loader/targets dispersos |
| `training/train_supervised.py` | Reemplazado por `training/train.py` |
| `training/validate_dataset.py` | Reemplazado y alineado al dataset definitivo |
| `training/evaluate.py` (vacío) | Implementado |
| `play_vs_alpha_zero.py` | Reemplazado por `gui/main.py` |
| `play_vs_ai.py`, `environment.py`, `agent/model.py`, `agent/policy.py`, `agent/train.py`, self-play y tests antiguos | Retirados del flujo final; eran parte del agente RL/acción 4,096 incompatible |
| `PythonChessGameEN.py`, `PythonChessGameSK.py` | Retirados; su lógica de reglas queda sustituida por `python-chess` |
| `pieces/`, `chess_pieces/` duplicadas | Una copia en `assets/pieces/` |
| Checkpoints `chess_policy*.pth` | No incluidos; arquitectura de política antigua incompatible |
| Datasets históricos, backups y self-play | No incluidos en `data/`; se mantienen únicamente en el ZIP original externo |

## 8. Siguiente paso recomendado

1. En el equipo RTX 4050, instalar `requirements.txt` y ejecutar el validador completo.
2. Continuar el entrenamiento con el mismo dataset y `--resume`.
3. Ejecutar un torneo con niveles Stockfish 2, 8 y 14, alternando colores y manteniendo las partidas truncadas separadas.
4. Comparar `--simulations 0`, `50` y `100` antes de fijar un valor predeterminado de MCTS.
5. Regenerar el dataset con el generador nuevo si se requiere una cobertura equilibrada de turnos y análisis Stockfish de mayor profundidad.
