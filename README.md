# DNLP – Chronos 2 Zero‑Shot Forecasting (Rossmann)

Python 3.10 required.

Pipeline per-negozio (no training, no cross-learning) con ablation delle context length e test di robustezza opzionali. I dati Rossmann sono giornalieri e includono covariate (Promo, holiday, ecc.).

## Struttura essenziale
```
main.py                     pipeline principale per tutti gli store
src/data/make_dataset.py    cleaning, continuità giornaliera, filtri store
src/models/predict_model.py inference Chronos 2
src/models/robustness.py    test di robustezza (noise, shuffle, ecc.)
src/evaluation/compare_results.py aggregazione WQL
src/evaluation/select_best_context.py riepilogo best context
src/visualization/generate_plots.py figure (solo store campione)
reports/                    report aggregati (wql, robustness, validity)
outputs/                    forecast per store e contesto (ctx_*)
auto_runs/                  script .bat/.sh per run end-to-end
```

## Cosa fa la pipeline
- Reindicizza ogni store su calendario giornaliero (no fill del target).
- Filtra store invalidi: run minimo continuo, finestra recente continua, covariate non nulle, soglia osservazioni, e filtro “zero tail” (run di zeri >10 o share zeri >20% nella finestra finale).
- Usa solo gli store validi per il loop.
- Ablation opzionale delle context length `[128, 256, 512]` (default solo 512).
- Salva forecast univariate/covariate per store in `outputs/ctx_<len>/`.
- Test di robustezza opzionali (noise, shuffle, missing future, ecc.) in `outputs/`.
- Aggrega WQL per store e per contesto in `reports/` (`wql_per_store.csv`, `wql_by_context.csv`, `wql_summary.csv`).
- Aggrega robustness in `reports/robustness_per_store_merged.csv` e `robustness_summary.csv`.
- Plots solo per pochi store campione (configurabile).

## Config chiave (in `main.py`)
- `RUN_ALL_CONTEXTS` (False di default): abilita ablation [128,256,512].
- `RUN_ROBUSTNESS` (False di default): esegue i test di robustezza.
- `SKIP_EXISTING_*`: metti a False per rigenerare tutto.
- Filtri store: `MIN_RUN`, `MIN_OBS`, `CHECK_RECENT_COVS`, `ZERO_TAIL_MAX`, `ZERO_TAIL_SHARE`.
- `CONTEXT_LENGTHS`: lista dei contesti (deriva da RUN_ALL_CONTEXTS).

## Come eseguire (one-click)
- Windows: `auto_runs\run_all.bat`
- mac/Linux: `bash auto_runs_mac/run_all.sh`

Oppure manuale:
```
python main.py
python src/evaluation/compare_results.py
python src/evaluation/select_best_context.py   # solo se più contesti
python src/visualization/generate_plots.py     # opzionale, store campione
```

## Output principali
- `reports/store_validity.csv`: dettaglio filtri per store (reasons, recent_window_ok, zero_tail_ok, ecc.).
- `reports/wql_per_store.csv` (+ `wql_by_context.csv`, `wql_summary.csv`): WQL aggregati.
- `reports/robustness_per_store_merged.csv`, `robustness_summary.csv`: aggregati robustness.
- `outputs/ctx_*/`: forecast per store (univariate/covariate) + ground truth.
- `outputs/comparison_report.txt`: riepilogo run/evaluation.

## Note su Robustness
Serve a misurare quanto le covariate contano (noise, shuffle, missing future, ecc.). È costoso: abilitalo solo se ti servono i grafici/CSV di stabilità.

## Plots
`generate_plots.py` genera figure solo per un campione di store (configurabile con `PLOT_SAMPLE_STORES` e `GENERATE_PER_STORE`). Questo evita migliaia di PNG.

## Zero-shot e niente training
Il modello `amazon/chronos-2` è usato solo in inference; nessun fine-tuning o cross-learning tra store. Filtri e continuità servono a mantenere coerente il setup paper-like.
