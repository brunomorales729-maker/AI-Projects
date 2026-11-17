# Changelog

## 2026-11-04 — Auditoría y consolidación

- Se creó una única estructura de proyecto, sin directorios V1/V2/V3, entrenamientos duplicados ni datasets históricos en el árbol principal.
- Se consolidaron 10,996 estados únicos de tres fuentes Stockfish en un único JSONL; los candidatos quedaron ordenados y con score desde el POV del jugador al turno.
- Se recuperaron `game_id` y `ply` para los 10,996 registros asociándolos a los JSONL originales. Las 245 secuencias fuente se unieron por transposiciones y quedaron en 241 grupos de partida.
- El split final es por partida completa: 9,258 posiciones de entrenamiento (208 grupos) y 1,738 de validación (33 grupos), sin intersección de grupos ni claves de posición.
- Se corrigió el generador para muestrear cada ply, ambos turnos y registrar `game_id`/`ply` en datasets futuros.
- Se añadió canonicalización relativa al jugador al turno y codificación explícita de promociones (20,480 acciones).
- Se conservó una ResNet Policy + Value de 64 canales y 4 bloques residuales (324,233 parámetros), compatible con el checkpoint supervisado original.
- Se implementó un loader compacto con targets dispersos; ya no retiene los diccionarios JSONL originales en memoria.
- Se añadieron AMP CUDA, gradient clipping, scheduler, early stopping, hash del dataset, reanudación y guardado atómico del único mejor checkpoint. Se maneja correctamente el cambio de checkpoint CPU (GradScaler vacío) a entrenamiento CUDA.
- Se conservó PUCT como opción configurable, no obligatoria; la inferencia normal no depende de Stockfish.
- Se implementó una GUI con elección de color, reinicio, movimientos legales, historial, promociones y simulaciones PUCT configurables.
- Se implementó evaluación contra Stockfish con CPL, tiempo por decisión, comparación opcional con checkpoint compatible y partidas truncadas excluidas del W/D/L.
- Se implementó el validador de estructura, grupos de partida, FEN/movimientos (validación completa cuando `python-chess` está instalado), duplicados, turnos y orden MultiPV.
- Pruebas en este entorno: 7/7 tests unitarios y compilación correctos. Se validó carga estricta del checkpoint y forward de la red.
- Se corrigió la deduplicación para incluir el contador de medio movimiento en la clave de estado, conservando 588 variantes relevantes para la regla de 50 movimientos.
- El dataset final contiene 10,996 estados, 51,357 evaluaciones candidatas, distribución por etapa 20.9/58.2/20.9 y distribución de turno 97.7% blancas / 2.3% negras.
- En el split final por partidas, el modelo ajustado (época 4) pasó de loss 8.0380 a 5.3009, top-1 de 2.70% a 5.58% y value MAE de 0.3027 a 0.2592 frente al checkpoint compatible original, evaluados en las mismas 1,738 posiciones.
- No se ejecutaron GUI, legalidad completa, torneo Stockfish ni CUDA/AMP porque faltaban `pygame-ce`, `python-chess`, el binario Stockfish y GPU CUDA en el entorno.
