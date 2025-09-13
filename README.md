# Chess Engine & Analysis Platform

UCI-compatible chess engine with bitboards, Zobrist hashing, alpha-beta (iterative deepening, TT, killers/history, quiescence, LMR, null-move, aspiration), and a tapered/NNUE-style evaluator.
FastAPI backend + React frontend for PV lines, eval graphs, and timings. NN training in PyTorch with self-play Elo tracking and SPRT gates.

## Layout
- engine/ : C++ engine (CMake)
- server/ : FastAPI backend (Python)
- nn/     : PyTorch training/export
- web/    : React UI (scaffold placeholder)
