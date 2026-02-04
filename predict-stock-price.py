import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import yfinance as yf
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler

    mo.md("# 📈 日本株 LSTM 株価予測（過去2か月 + 未来1か月）")

    ticker_ui = mo.ui.text(label="日本株コード（例: 7203, 9984, 1306.T）", value="7203")

    window_ui = mo.ui.number(label="Train window size", value=20, step=1)

    epochs_ui = mo.ui.number(label="Epochs", value=20, step=1)

    mo.hstack([ticker_ui, window_ui, epochs_ui])
    return (
        MinMaxScaler,
        epochs_ui,
        mo,
        nn,
        np,
        pd,
        plt,
        ticker_ui,
        torch,
        window_ui,
        yf,
    )


@app.cell
def _(torch):
    from pandas.tseries.offsets import BDay

    def normalize_jp_ticker(ticker: str) -> str:
        t = ticker.strip().upper()
        if t.isdigit():
            return f"{t}.T"
        return t

    def make_sequences(series, window):
        X, y = [], []
        for i in range(len(series) - window):
            X.append(series[i : i + window])
            y.append(series[i + window])
        return torch.stack(X), torch.stack(y)

    return BDay, make_sequences, normalize_jp_ticker


@app.cell
def _(
    MinMaxScaler,
    epochs_ui,
    make_sequences,
    mo,
    normalize_jp_ticker,
    ticker_ui,
    torch,
    window_ui,
    yf,
):
    ticker = normalize_jp_ticker(ticker_ui.value)
    window = int(window_ui.value)
    epochs = int(epochs_ui.value)

    df = yf.download(ticker, start="2022-01-01", progress=False)

    if df.empty:
        mo.md(f"❌ データ取得失敗: {ticker}")
        raise SystemExit

    df = df.sort_index()[["Close"]]

    name = yf.Ticker(ticker).info.get("longName", "Unknown")
    mo.md(f"## {name} ({ticker})")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(df["Close"].values.reshape(-1, 1))
    series = torch.FloatTensor(scaled).view(-1)

    X, y = make_sequences(series, window)

    mo.md(f"""
    - データ数: {len(df)}
    - 学習サンプル数: {len(X)}
    """)
    return X, df, epochs, name, scaler, series, ticker, window, y


@app.cell
def _(X, epochs, mo, nn, torch, y):
    class LSTM(nn.Module):
        def __init__(self, hidden=50):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, seq):
            h0 = torch.zeros(1, 1, self.lstm.hidden_size)
            c0 = torch.zeros(1, 1, self.lstm.hidden_size)
            out, _ = self.lstm(seq.view(len(seq), 1, 1), (h0, c0))
            return self.fc(out[-1]).squeeze(0)

    model = LSTM()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for e in range(epochs):
        total = 0.0
        for xs, ys in zip(X, y):
            optimizer.zero_grad()
            pred = model(xs)
            loss = loss_fn(pred, ys.view(1))
            loss.backward()
            optimizer.step()
            total += loss.item()
        mo.md(f"Epoch {e + 1}/{epochs} | loss={total / len(X):.6f}")

    mo.md("✅ Training 完了")
    return (model,)


@app.cell
def _(
    BDay,
    df,
    model,
    name,
    np,
    pd,
    plt,
    scaler,
    series,
    ticker,
    torch,
    window,
):
    model.eval()

    # --- 過去再現 ---
    past_preds = []
    with torch.no_grad():
        for i in range(len(series) - window):
            past_preds.append(model(series[i : i + window]).item())

    past_preds = scaler.inverse_transform(np.array(past_preds).reshape(-1, 1))

    # --- 未来1か月 ---
    future_days = 20
    future_preds = []
    current = series[-window:].clone()

    with torch.no_grad():
        for _ in range(future_days):
            p = model(current)
            future_preds.append(p.item())
            current = torch.cat((current[1:], p))

    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

    # --- 日付 ---
    past_days = 60
    past_actual = df["Close"].iloc[-past_days:]
    past_pred = past_preds[-past_days:]
    past_dates = df.index[-past_days:]

    future_dates = pd.bdate_range(start=df.index[-1] + BDay(1), periods=future_days)

    # --- Plot ---
    fig = plt.figure(figsize=(12, 6))

    plt.plot(
        past_actual.index, past_actual.values, label="Actual (Past 2 Months)", lw=2
    )
    plt.plot(past_dates, past_pred, "--", label="Predicted (Past)")
    plt.plot(future_dates, future_preds, ":", lw=2, label="Predicted (Future 1 Month)")

    plt.title(f"{name} ({ticker})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    fig
    return


if __name__ == "__main__":
    app.run()
