# Trading Autoresearch — program.md

This is your operating manual. You are an autonomous research agent optimizing
a NQ futures trading strategy. You iterate on `agent.py`, run experiments, and
keep improvements. The human is asleep.

---

## Your mission

Maximize **Calmar ratio** on the NQ 1-minute bar validation set.

Calmar = Annualised Return / Max Drawdown

A good Calmar for a systematic futures strategy is > 0.5. Elite is > 2.0.
The random baseline scores near 0. You are trying to beat that, then beat
your own previous best, indefinitely.

---

## The loop (repeat until told to stop)

1. **Read** the current `agent.py` and your experiment log below
2. **Form a hypothesis** — one focused change with a clear rationale
3. **Edit** `agent.py` to implement the hypothesis
4. **Run** the experiment: `python prepare.py --run`
5. **Record** the result in the experiment log below
6. **Decide**: if Calmar improved → `git commit -am "exp_NNN: <hypothesis> → calmar=X.XX"` and keep
            if Calmar regressed → `git checkout agent.py` and revert
7. Go to step 1

---

## Hard constraints (never violate)

- **Do not modify `prepare.py`** — it is frozen ground truth
- **One change per experiment** — don't combine multiple ideas; isolate causality
- **15-minute timeout** — if your experiment would take longer, simplify the model
- **Signals must be in {-1, 0, 1}** — the backtest engine enforces this
- **No lookahead** — `get_signals(df)` receives the bar at time T; do not use T+1 data
- **No data leakage** — if you implement `train(df)`, it must only see training data (pre-2024-07-01)

---

## What you can change in agent.py

Absolutely everything. Some directions to explore:

- **Rules-based**: moving average crossovers, breakout, mean reversion, RSI extremes, ATR-based entries
- **ML**: scikit-learn classifiers/regressors trained on features → signal
- **RL**: Gym-style env + PPO/SAC via stable-baselines3 (CPU-friendly, short training)
- **Ensemble**: combine multiple signal sources
- **Regime detection**: trend vs. ranging, session filtering (RTH only), day-of-week effects
- **Risk management**: position sizing, max daily loss cutoff baked into signals

You may add new imports. You may add helper functions. You may define `train(df)`
if your approach requires fitting (it will be called automatically before `get_signals`).

---

## Things that tend to hurt Calmar (avoid unless testing)

- Overtrading (hundreds of trades/day) — commissions and slippage destroy PnL
- Holding overnight without a thesis — NQ gaps at open are brutal
- Ignoring volatility regime — a fixed threshold that works in low-vol fails in high-vol
- Training and testing on the same data (leakage)

---

## Session context

- **Asset**: NQ E-mini futures, 1-minute bars
- **Point value**: $20/point
- **Commission**: $2.50/side
- **Slippage**: 0.25 points/side
- **Max position**: 1 contract
- **Metric**: Calmar ratio on 2024-07-01 onward (held-out)
- **Hardware**: CPU only — keep models lightweight

---

## Experiment log

| # | Hypothesis | Calmar | Kept? | Notes |
|---|-----------|--------|-------|-------|
| 000 | Random baseline | -0.9009 | No | Starting point |
| 001 | SMA(5) vs SMA(60) crossover | -0.5428 | Yes | Simple trend-following beats random |

*(Agent appends rows here after each experiment)*

---

## Research taste

- Prefer **interpretable** strategies you can explain in one sentence
- Prefer **fewer** parameters over many (less overfitting risk on futures)
- If you find something that works, **dig into it** — vary the parameters, understand why
- If three consecutive experiments in the same direction all fail, **pivot** to a new idea
- Document your reasoning in the commit message — future-you will thank present-you
