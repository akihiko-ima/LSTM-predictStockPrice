# LSTM-predict stock price

株価の予測を行う予測モデルの作成に理解を深めるため、[こちら](https://www.udemy.com/course/pythonai/?couponCode=KEEPLEARNING)
を参考にソースコードを実際に書いて動作を確認した。

## marimo commands

```bash
uv run marimo edit predict-stock-price.py
```

```bash
uv run marimo run predict-stock-price.py
```

```bash
uv run marimo convert your_notebook.ipynb -o your_notebook.py
```

#### タスクランナー ruff & mypy

- ruff の導入

```bash
uv add --dev ruff
```

- mypy の導入

```bash
uv add --dev mypy
```

- タスクランナーの作成

```bash
uv add --dev poethepoet
```

```bash
# pyproject.toml に追記

[tool.poe.tasks]
format = "uv run ruff format ."
lint = "uv run ruff check --fix ."
type-check = "uv run mypy ."

check = ["format", "lint", "type-check"]
```

- poe (ruff と mypy をすべて実行)

```bash
uv run poe check
```
