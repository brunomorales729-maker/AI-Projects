# ChessAI — agente neuronal supervisado por Stockfish

Proyecto compacto de ajedrez con una sola red neuronal Policy + Value, un dataset definitivo generado a partir de Stockfish y una interfaz gráfica PyGame. Stockfish es **profesor de datos** y rival de evaluación; la inferencia normal de la IA no lo inicia ni lo consulta.

## Arquitectura

```text
ChessAI/
├── agent/
│   ├── encoding.py       # 18 planos y 20,480 acciones con promociones
│   ├── network.py        # ResNet residual + Policy Head + Value Head
│   ├── policy.py         # Inferencia neuronal y máscara de movimientos legales
│   └── mcts.py           # PUCT opcional guiado por la red
├── training/
│   ├── generate_dataset.py
│   ├── validate_dataset.py
│   ├── dataset.py        # Loader compacto y targets dispersos
│   ├── train.py          # Único entrenamiento supervisado
│   └── evaluate.py       # Partidas contra Stockfish y evaluación de decisiones
├── data/stockfish_dataset.jsonl
├── models/chess_model_best.pth
├── gui/main.py
├── assets/pieces/        # Un juego de 12 piezas SVG
├── tests/test_core.py
├── config.py
└── requirements.txt
```

## Instalación en Ubuntu

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
sudo apt install stockfish
export STOCKFISH_EXECUTABLE="$(command -v stockfish)"
```

Comprueba PyTorch y CUDA:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

La interfaz y la inferencia no necesitan Stockfish; solo se requiere para generar datos y evaluar partidas.

## Validar el dataset

```bash
python -m training.validate_dataset
```

El dataset incluido contiene 10,996 posiciones Stockfish, 51,357 evaluaciones candidatas, 241 grupos de partida, evaluaciones desde la perspectiva del jugador al turno y metadatos `game_id`/`ply`. Con `python-chess` instalado, el validador comprueba también legalidad de posiciones y movimientos candidatos.

## Generar un dataset nuevo

El generador muestrea cada ply, incluye ambos turnos, controla cuotas por etapa y escribe `game_id` y `ply` para permitir separación por partida. No sobrescribe el dataset por defecto; `--overwrite` hace una copia externa antes de reemplazarlo.

```bash
python -m training.generate_dataset \
  --engine "$STOCKFISH_EXECUTABLE" \
  --positions 10000 \
  --multipv 5 \
  --analysis-time 0.06 \
  --play-time 0.01 \
  --overwrite
```

Si no alcanza las cuotas, descarta el archivo temporal y conserva el dataset anterior.

## Entrenar

Para continuar el checkpoint incluido:

```bash
python -m training.train --resume --epochs 40 --batch-size 32
```

Para entrenar desde pesos nuevos, omite `--resume`. Para inicializar desde otro checkpoint compatible Policy + Value, usa `--warm-start ruta/al/checkpoint.pth`.

El split se hace por `game_id` (partidas completas) y comprueba que no haya claves de posición compartidas. El entrenamiento usa AdamW, gradient clipping, scheduler, early stopping, checkpoint atómico y mixed precision en CUDA. El mejor modelo se guarda en `models/chess_model_best.pth`; el SHA-256 del dataset queda dentro del checkpoint y `--resume` rechaza datos distintos.

Para la RTX 4050 de 6 GB empieza con batch 32; si CUDA reporta falta de memoria, prueba 16 u 8. El checkpoint incluido es un ajuste supervisado inicial (mejor época guardada: 4), no una convergencia final.

## Jugar

```bash
python -m gui.main
```

La interfaz permite elegir blancas o negras, reiniciar, ver movimientos legales e historial, elegir promoción con Q/R/B/N y ajustar simulaciones PUCT. `0` simulaciones usa directamente la red. La IA no consulta Stockfish durante la partida.

## Evaluar

```bash
python -m training.evaluate \
  --stockfish "$STOCKFISH_EXECUTABLE" \
  --skill-levels 2,8,14 \
  --games-per-level 4 \
  --simulations 50 \
  --quality-depth 8 \
  --output evaluation.json
```

Las partidas que alcanzan `--max-plies` se marcan como `truncated`; no se cuentan como tablas ni entran al win rate. El informe registra W/D/L, tiempo por decisión, pérdida centipawn y coincidencia con la mejor jugada de Stockfish por etapa. Se puede comparar un checkpoint compatible con `--baseline-checkpoint ruta.pth`.

## Pruebas

```bash
python -m unittest discover -s tests -v
python -m compileall -q agent training gui tests
```

La evaluación supervisada del checkpoint ajustado mejoró frente al original en un split reservado por partidas (loss 8.03795 → 5.30090; top-1 2.70% → 5.58%; value MAE 0.30272 → 0.25923). Esto no demuestra Elo ni fuerza competitiva: se debe ejecutar el torneo contra Stockfish en el equipo objetivo.
