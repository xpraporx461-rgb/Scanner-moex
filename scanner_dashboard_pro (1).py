import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Сканер PRO — дейтрейдинг", page_icon="🚀", layout="wide")

MOEX_STOCKS = ["SBER", "GAZP", "LKOH", "YNDX", "MTSS", "MGNT", "ROSN", "NVTK",
               "TATN", "GMKN", "PLZL", "ALRS", "CHMF", "SNGS", "SNGSP", "NLMK",
               "MAGN", "VTBR"]
MOEX_FUTURES = ["SiM5", "RIM5", "GZM5", "SRM5", "MXM5", "EDM5", "CRM5", "VXM5", "BRM5"]

# ============ DATA ============
def get_mock_data(ticker, days=60):
    np.random.seed(hash(ticker) % 2**32)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq="B")
    close = 100 + np.cumsum(np.random.randn(days) * 2)
    high = close + np.abs(np.random.randn(days) * 1.5)
    low = close - np.abs(np.random.randn(days) * 1.5)
    open_ = close + np.random.randn(days) * 0.5
    volume = np.random.randint(100000, 1000000, days).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume}, index=dates)

def get_moex_data(ticker, days=60, is_future=False):
    try:
        import aiomoex
        engine = "futures" if is_future else "stock"
        board = "RFUD" if is_future else "TQBR"
        candles = aiomoex.get_candles(ticker, engine, board, interval="day", n=days)
        df = pd.DataFrame(candles)
        df["date"] = pd.to_datetime(df["begin"])
        df.set_index("date", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return get_mock_data(ticker, days)

def get_tinkoff_data(ticker, token, interval="1min", days=1, is_future=False):
    try:
        from tinkoff.invest import AsyncClient, CandleInterval
        import asyncio
        from datetime import datetime, timedelta, timezone
        interval_map = {"1min": CandleInterval.CANDLE_INTERVAL_1_MIN,
                        "5min": CandleInterval.CANDLE_INTERVAL_5_MIN,
                        "10min": CandleInterval.CANDLE_INTERVAL_10_MIN,
                        "1hour": CandleInterval.CANDLE_INTERVAL_HOUR,
                        "1day": CandleInterval.CANDLE_INTERVAL_DAY}
        async def fetch():
            async with AsyncClient(token) as client:
                end = datetime.now(timezone.utc)
                start = end - timedelta(days=days)
                resp = await client.get_candles(figi=ticker, from_=start, to=end,
                                                interval=interval_map[interval])
                candles = []
                for c in resp.candles:
                    candles.append({"open": float(c.open.units) + c.open.nano / 1e9,
                                  "high": float(c.high.units) + c.high.nano / 1e9,
                                  "low": float(c.low.units) + c.low.nano / 1e9,
                                  "close": float(c.close.units) + c.close.nano / 1e9,
                                  "volume": float(c.volume), "date": c.time})
                return pd.DataFrame(candles).set_index("date")
        return asyncio.run(fetch())
    except Exception as e:
        st.warning(f"Tinkoff API недоступен ({e}). Использую mock-данные.")
        return get_mock_data(ticker, 60)

# ============ INDICATORS ============
def add_indicators(df, ma_s=20, ma_l=50, ma_xl=200, rsi_p=14, atr_p=14):
    df = df.copy()
    df["MA_S"] = df["close"].rolling(ma_s).mean()
    df["MA_L"] = df["close"].rolling(ma_l).mean()
    period = min(ma_xl, len(df))
    df["MA_XL"] = df["close"].rolling(period).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_p).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_p).mean()
    rs = gain / loss.replace(0, 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["ATR"] = (df["high"] - df["low"]).rolling(atr_p).mean()
    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["high_20"] = df["high"].rolling(20).max()
    df["low_20"] = df["low"].rolling(20).min()
    return df

# ============ STRATEGIES ============
def filter_breakout(df, threshold=0.005, vol_ratio=1.3):
    last = df.iloc[-1]
    level = last["high_20"] * (1 + threshold)
    if last["close"] > level and last["volume"] > vol_ratio * last["vol_ma"]:
        atr = last["ATR"]
        return {"strategy": "Breakout", "direction": "LONG",
                "entry": last["close"], "stop": last["close"] - atr,
                "target": last["close"] + 2*atr, "atr": round(atr, 2),
                "rsi": round(last["RSI"], 1),
                "vol_ratio": round(last["volume"]/last["vol_ma"], 2)}
    return None

def filter_range(df, lower=30, upper=40):
    last = df.iloc[-1]
    width = last["high_20"] - last["low_20"]
    if width > 0:
        dist = (last["close"] - last["low_20"]) / width
        if dist < 0.15 and lower <= last["RSI"] <= upper:
            return {"strategy": "Range", "direction": "LONG",
                    "entry": last["close"], "stop": last["low_20"],
                    "target": last["high_20"], "atr": round(last["ATR"], 2),
                    "rsi": round(last["RSI"], 1),
                    "vol_ratio": round(last["volume"]/last["vol_ma"], 2)}
    return None

def filter_pullback(df, tolerance=0.015):
    last = df.iloc[-1]
    if last["close"] > last["MA_XL"] and abs(last["close"]-last["MA_S"])/last["MA_S"] < tolerance:
        atr = last["ATR"]
        return {"strategy": "Pullback", "direction": "LONG",
                "entry": last["close"], "stop": last["close"] - atr,
                "target": last["close"] + 2*atr, "atr": round(atr, 2),
                "rsi": round(last["RSI"], 1),
                "vol_ratio": round(last["volume"]/last["vol_ma"], 2)}
    return None

def filter_gap(df, gap_pct=0.02):
    last = df.iloc[-1]; prev = df.iloc[-2]
    gap = (last["open"] - prev["close"]) / prev["close"]
    if abs(gap) > gap_pct:
        atr = last["ATR"]
        direction = "LONG" if gap > 0 else "SHORT"
        stop = last["close"] - atr if direction == "LONG" else last["close"] + atr
        target = last["close"] + 2*atr if direction == "LONG" else last["close"] - 2*atr
        return {"strategy": "Gap", "direction": direction,
                "entry": last["close"], "stop": stop, "target": target,
                "atr": round(atr, 2), "rsi": round(last["RSI"], 1),
                "vol_ratio": round(last["volume"]/last["vol_ma"], 2)}
    return None

def filter_volume_spike(df, vol_mult=3.0, price_pct=0.015):
    last = df.iloc[-1]
    if last["volume"] > vol_mult * last["vol_ma"]:
        change = (last["close"] - last["open"]) / last["open"]
        if abs(change) > price_pct:
            atr = last["ATR"]
            direction = "LONG" if change > 0 else "SHORT"
            stop = last["close"] - atr if direction == "LONG" else last["close"] + atr
            target = last["close"] + 2*atr if direction == "LONG" else last["close"] - 2*atr
            return {"strategy": "Volume Spike", "direction": direction,
                    "entry": last["close"], "stop": stop, "target": target,
                    "atr": round(atr, 2), "rsi": round(last["RSI"], 1),
                    "vol_ratio": round(last["volume"]/last["vol_ma"], 2)}
    return None

STRATEGIES = {"Breakout": filter_breakout, "Range": filter_range,
              "Pullback": filter_pullback, "Gap": filter_gap,
              "Volume Spike": filter_volume_spike}

# ============ SCAN ============
def scan_tickers(tickers, selected_strategies, is_future, params, token=None):
    results = []
    progress = st.progress(0)
    for i, t in enumerate(tickers):
        try:
            if token:
                df = get_tinkoff_data(t, token, interval="5min", days=1, is_future=is_future)
            else:
                df = get_moex_data(t, is_future=is_future)
            df = add_indicators(df, ma_s=params["ma_s"], ma_l=params["ma_l"],
                              rsi_p=params["rsi_p"], atr_p=params["atr_p"])
            for name in selected_strategies:
                func = STRATEGIES[name]
                if name == "Breakout":
                    sig = func(df, threshold=params["breakout_threshold"], vol_ratio=params["breakout_vol"])
                elif name == "Range":
                    sig = func(df, lower=params["rsi_low"], upper=params["rsi_high"])
                elif name == "Pullback":
                    sig = func(df, tolerance=params["pullback_tol"])
                elif name == "Gap":
                    sig = func(df, gap_pct=params["gap_pct"])
                elif name == "Volume Spike":
                    sig = func(df, vol_mult=params["vol_spike_mult"], price_pct=params["vol_price_pct"])
                if sig:
                    sig["ticker"] = t
                    results.append(sig)
        except Exception as e:
            st.warning(f"Ошибка по {t}: {e}")
        progress.progress((i+1)/len(tickers))
    return pd.DataFrame(results)

# ============ BACKTEST ============
def run_backtest(df, strategy_func, params, capital=100000, risk_pct=1.0, lot_size=1):
    trades = []
    equity = [capital]
    position = None
    for i in range(50, len(df)):
        window = df.iloc[:i+1].copy()
        window = add_indicators(window, ma_s=params["ma_s"], ma_l=params["ma_l"],
                              rsi_p=params["rsi_p"], atr_p=params["atr_p"])
        if strategy_func.__name__ == "filter_breakout":
            sig = strategy_func(window, threshold=params["breakout_threshold"], vol_ratio=params["breakout_vol"])
        elif strategy_func.__name__ == "filter_range":
            sig = strategy_func(window, lower=params["rsi_low"], upper=params["rsi_high"])
        elif strategy_func.__name__ == "filter_pullback":
            sig = strategy_func(window, tolerance=params["pullback_tol"])
        elif strategy_func.__name__ == "filter_gap":
            sig = strategy_func(window, gap_pct=params["gap_pct"])
        elif strategy_func.__name__ == "filter_volume_spike":
            sig = strategy_func(window, vol_mult=params["vol_spike_mult"], price_pct=params["vol_price_pct"])
        else:
            sig = None

        if position is not None:
            row = window.iloc[-1]
            if position["direction"] == "LONG":
                if row["low"] <= position["stop"]:
                    pnl = (position["stop"] - position["entry"]) * position["lots"] * lot_size
                    trades.append({"entry_date": position["date"], "exit_date": window.index[-1],
                                  "entry": position["entry"], "exit": position["stop"],
                                  "direction": "LONG", "pnl": round(pnl, 2), "result": "LOSS"})
                    equity.append(equity[-1] + pnl)
                    position = None
                elif row["high"] >= position["target"]:
                    pnl = (position["target"] - position["entry"]) * position["lots"] * lot_size
                    trades.append({"entry_date": position["date"], "exit_date": window.index[-1],
                                  "entry": position["entry"], "exit": position["target"],
                                  "direction": "LONG", "pnl": round(pnl, 2), "result": "WIN"})
                    equity.append(equity[-1] + pnl)
                    position = None
            else:
                if row["high"] >= position["stop"]:
                    pnl = (position["entry"] - position["stop"]) * position["lots"] * lot_size
                    trades.append({"entry_date": position["date"], "exit_date": window.index[-1],
                                  "entry": position["entry"], "exit": position["stop"],
                                  "direction": "SHORT", "pnl": round(pnl, 2), "result": "LOSS"})
                    equity.append(equity[-1] + pnl)
                    position = None
                elif row["low"] <= position["target"]:
                    pnl = (position["entry"] - position["target"]) * position["lots"] * lot_size
                    trades.append({"entry_date": position["date"], "exit_date": window.index[-1],
                                  "entry": position["entry"], "exit": position["target"],
                                  "direction": "SHORT", "pnl": round(pnl, 2), "result": "WIN"})
                    equity.append(equity[-1] + pnl)
                    position = None

        if position is None and sig:
            risk_amount = equity[-1] * risk_pct / 100
            stop_distance = abs(sig["entry"] - sig["stop"])
            if stop_distance > 0 and sig["atr"] > 0:
                lots = max(1, int(risk_amount / (stop_distance * lot_size)))
                position = {"entry": sig["entry"], "stop": sig["stop"],
                           "target": sig["target"], "direction": sig["direction"],
                           "lots": lots, "date": window.index[-1]}

    return trades, equity

# ============ POSITION SIZING ============
def calc_position_size(entry, stop, target, capital, risk_pct, lot_size=1):
    risk_amount = capital * risk_pct / 100
    stop_distance = abs(entry - stop)
    if stop_distance == 0:
        return None
    lots = max(1, int(risk_amount / (stop_distance * lot_size)))
    position_value = lots * entry * lot_size
    actual_risk = stop_distance * lots * lot_size
    potential_profit = abs(target - entry) * lots * lot_size
    leverage = position_value / capital if capital > 0 else 0
    rr_ratio = potential_profit / actual_risk if actual_risk > 0 else 0
    if risk_pct <= 1:
        risk_label = "✅ Безопасно"
    elif risk_pct <= 3:
        risk_label = "⚠️ Умеренно"
    else:
        risk_label = "🔴 Опасно"
    return {"lots": lots, "position_value": round(position_value, 2),
            "actual_risk": round(actual_risk, 2),
            "potential_profit": round(potential_profit, 2),
            "leverage": round(leverage, 2), "rr_ratio": round(rr_ratio, 2),
            "risk_label": risk_label, "risk_amount": round(risk_amount, 2)}

# ============ UI ============
st.title("🚀 Сканер PRO — дейтрейдинг MOEX")

mode = st.sidebar.selectbox("Режим работы",
    ["📊 Сканер сигналов", "📈 Бэктест", "🎯 Размер позиции", "⏱️ Real-time"])

market = st.sidebar.radio("Рынок", ["Акции MOEX (TQBR)", "Фьючерсы FORTS"])
is_future = "Фьючерсы" in market
default_tickers = MOEX_FUTURES if is_future else MOEX_STOCKS
tickers_str = st.sidebar.text_area("Тикеры (через запятую)", ", ".join(default_tickers))
tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

st.sidebar.subheader("Стратегии")
selected_strategies = st.sidebar.multiselect("Выберите стратегии",
    list(STRATEGIES.keys()), list(STRATEGIES.keys()))

st.sidebar.subheader("Параметры индикаторов")
params = {
    "ma_s": st.sidebar.slider("MA короткая", 5, 50, 20),
    "ma_l": st.sidebar.slider("MA длинная", 20, 100, 50),
    "rsi_p": st.sidebar.slider("RSI период", 5, 30, 14),
    "atr_p": st.sidebar.slider("ATR период", 5, 30, 14),
    "breakout_threshold": st.sidebar.slider("Порог пробоя %", 0.1, 3.0, 0.5, 0.1) / 100,
    "breakout_vol": st.sidebar.slider("Объём пробоя x среднего", 1.0, 3.0, 1.3, 0.1),
    "rsi_low": st.sidebar.slider("RSI перепроданность", 10, 40, 30),
    "rsi_high": st.sidebar.slider("RSI перекупленность", 50, 80, 40),
    "pullback_tol": st.sidebar.slider("Толерантность к MA %", 0.5, 5.0, 1.5, 0.1) / 100,
    "gap_pct": st.sidebar.slider("Гэп %", 0.5, 5.0, 2.0, 0.1) / 100,
    "vol_spike_mult": st.sidebar.slider("Объём спайк x среднего", 1.5, 10.0, 3.0, 0.5),
    "vol_price_pct": st.sidebar.slider("Движение цены для спайка %", 0.5, 5.0, 1.5, 0.1) / 100,
}

# --- SCANNER ---
if mode == "📊 Сканер сигналов":
    if st.sidebar.button("🔍 Запустить сканирование", type="primary"):
        with st.spinner("Сканирование..."):
            signals = scan_tickers(tickers, selected_strategies, is_future, params)
        if signals.empty:
            st.info("Сигналов не найдено.")
        else:
            st.subheader(f"📋 Сигналов: {len(signals)}")
            display_cols = ["ticker", "strategy", "direction", "entry", "stop", "target", "atr", "rsi", "vol_ratio"]
            display_df = signals[display_cols].rename(columns={
                "ticker": "Тикер", "strategy": "Стратегия", "direction": "Направление",
                "entry": "Вход", "stop": "Стоп", "target": "Тейк",
                "atr": "ATR", "rsi": "RSI", "vol_ratio": "Объём x"})
            st.dataframe(display_df, use_container_width=True)
            csv = signals.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Скачать CSV", csv, "signals.csv", "text/csv")

# --- BACKTEST ---
elif mode == "📈 Бэктест":
    st.subheader("📈 Бэктест стратегий")
    bt_ticker = st.selectbox("Тикер для бэктеста", tickers)
    bt_strategy_name = st.selectbox("Стратегия", list(STRATEGIES.keys()))
    capital = st.number_input("Стартовый капитал (руб)", 10000, 10000000, 100000, 10000)
    risk_pct = st.slider("Риск на сделку %", 0.5, 10.0, 1.0, 0.5)
    lot_size = st.number_input("Размер лота", 1, 1000, 1)

    if st.button("▶️ Запустить бэктест", type="primary"):
        with st.spinner("Бэктест..."):
            df = get_moex_data(bt_ticker, is_future=is_future)
            df = add_indicators(df, ma_s=params["ma_s"], ma_l=params["ma_l"],
                              rsi_p=params["rsi_p"], atr_p=params["atr_p"])
            trades, equity = run_backtest(df, STRATEGIES[bt_strategy_name], params,
                                         capital, risk_pct, lot_size)
        if not trades:
            st.warning("Сделок не было. Попробуйте изменить параметры.")
        else:
            trades_df = pd.DataFrame(trades)
            wins = trades_df[trades_df["result"] == "WIN"]
            losses = trades_df[trades_df["result"] == "LOSS"]
            total_pnl = trades_df["pnl"].sum()
            win_rate = len(wins) / len(trades_df) * 100
            gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
            gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
            equity_series = pd.Series(equity)
            max_dd = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max() * 100

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Win Rate", f"{win_rate:.1f}%")
            col2.metric("Profit Factor", f"{profit_factor:.2f}")
            col3.metric("Total P&L", f"{total_pnl:,.0f} руб", delta=f"{total_pnl/capital*100:.1f}%")
            col4.metric("Max Drawdown", f"{max_dd:.1f}%")

            col5, col6 = st.columns(2)
            avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
            avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
            col5.metric("Avg Win", f"{avg_win:,.0f} руб")
            col6.metric("Avg Loss", f"{avg_loss:,.0f} руб")

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(y=equity, mode="lines", name="Capital",
                            line=dict(color="blue", width=2)))
            fig_eq.update_layout(title="Кривая капитала", xaxis_title="Сделка",
                               yaxis_title="Капитал")
            st.plotly_chart(fig_eq, use_container_width=True)

            st.subheader("📋 История сделок")
            st.dataframe(trades_df, use_container_width=True)
            csv = trades_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Скачать сделки CSV", csv, "backtest_trades.csv", "text/csv")

# --- POSITION SIZING ---
elif mode == "🎯 Размер позиции":
    st.subheader("🎯 Калькулятор размера позиции")
    col_in, col_stop, col_target = st.columns(3)
    with col_in:
        entry_price = st.number_input("Цена входа", 0.01, 100000.0, 100.0, 0.01)
    with col_stop:
        stop_price = st.number_input("Цена стопа", 0.01, 100000.0, 95.0, 0.01)
    with col_target:
        target_price = st.number_input("Цена тейка", 0.01, 100000.0, 110.0, 0.01)

    col_cap, col_risk, col_lot = st.columns(3)
    with col_cap:
        capital = st.number_input("Капитал (руб)", 1000, 10000000, 100000, 1000)
    with col_risk:
        risk_pct = st.slider("Риск на сделку %", 0.5, 10.0, 1.0, 0.5)
    with col_lot:
        lot_size = st.number_input("Размер лота", 1, 1000, 1)

    if st.button("Рассчитать", type="primary"):
        result = calc_position_size(entry_price, stop_price, target_price, capital, risk_pct, lot_size)
        if result:
            st.markdown(f"### {result['risk_label']}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Количество лотов", result["lots"])
            col2.metric("Стоимость позиции", f"{result['position_value']:,.0f} руб")
            col3.metric("Риск в деньгах", f"{result['actual_risk']:,.0f} руб")
            col4, col5, col6 = st.columns(3)
            col4.metric("Потенциальная прибыль", f"{result['potential_profit']:,.0f} руб")
            col5.metric("Кредитное плечо", f"{result['leverage']:.2f}x")
            col6.metric("Риск/Прибыль", f"1:{result['rr_ratio']:.1f}")

            st.subheader("📋 Сценарии")
            scenarios = pd.DataFrame([
                {"Сценарий": "WIN ✅", "Цена выхода": target_price,
                 "P&L на лот": round(abs(target_price - entry_price) * lot_size, 2),
                 "P&L всего": result["potential_profit"]},
                {"Сценарий": "LOSS ❌", "Цена выхода": stop_price,
                 "P&L на лот": round(-abs(entry_price - stop_price) * lot_size, 2),
                 "P&L всего": -result["actual_risk"]},
                {"Сценарий": "Безубыток ➖", "Цена выхода": entry_price,
                 "P&L на лот": 0.0, "P&L всего": 0.0},
            ])
            st.dataframe(scenarios, use_container_width=True)

            fig = go.Figure()
            fig.add_hrect(y0=stop_price, y1=entry_price, fillcolor="red", opacity=0.2,
                         annotation_text=f"Зона риска ({result['actual_risk']:,.0f} руб)")
            fig.add_hrect(y0=entry_price, y1=target_price, fillcolor="green", opacity=0.2,
                         annotation_text=f"Зона прибыли ({result['potential_profit']:,.0f} руб)")
            fig.add_hline(y=entry_price, line_color="blue", annotation_text="Вход")
            fig.add_hline(y=stop_price, line_dash="dash", line_color="red", annotation_text="Стоп")
            fig.add_hline(y=target_price, line_dash="dash", line_color="green", annotation_text="Тейк")
            fig.update_layout(title="Зоны риска и прибыли", yaxis_title="Цена",
                            xaxis={"visible": False}, height=400)
            st.plotly_chart(fig, use_container_width=True)

# --- REALTIME ---
elif mode == "⏱️ Real-time":
    st.subheader("⏱️ Real-time данные (Tinkoff Invest API)")
    token = st.text_input("API токен Tinkoff Invest", type="password",
                         placeholder="Введите токен из приложения Тинькофф Инвестиции")
    rt_ticker = st.selectbox("Тикер", tickers)
    rt_interval = st.selectbox("Интервал", ["1min", "5min", "10min", "1hour", "1day"])
    rt_days = st.slider("Глубина истории (дней)", 1, 30, 1)

    if st.button("🔄 Обновить данные", type="primary") and token:
        with st.spinner("Загрузка..."):
            df = get_tinkoff_data(rt_ticker, token, interval=rt_interval, days=rt_days, is_future=is_future)
        if df is not None and len(df) > 0:
            df = add_indicators(df, ma_s=params["ma_s"], ma_l=params["ma_l"],
                              rsi_p=params["rsi_p"], atr_p=params["atr_p"])
            last = df.iloc[-1]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Последняя цена", f"{last['close']:.2f}")
            col2.metric("RSI", f"{last['RSI']:.1f}")
            col3.metric("ATR", f"{last['ATR']:.2f}")
            col4.metric("Объём", f"{last['volume']:,.0f}")

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               vertical_spacing=0.08, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"],
                        low=df["low"], close=df["close"], name="Цена"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MA_S"], name="MA краткая",
                        line=dict(width=1, color="blue")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MA_L"], name="MA длинная",
                        line=dict(width=1, color="orange")), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Объём",
                        marker_color="gray", opacity=0.5), row=2, col=1)
            fig.update_layout(height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Сигналы по последнему бару")
            rt_signals = []
            for name in selected_strategies:
                func = STRATEGIES[name]
                if name == "Breakout":
                    sig = func(df, threshold=params["breakout_threshold"], vol_ratio=params["breakout_vol"])
                elif name == "Range":
                    sig = func(df, lower=params["rsi_low"], upper=params["rsi_high"])
                elif name == "Pullback":
                    sig = func(df, tolerance=params["pullback_tol"])
                elif name == "Gap":
                    sig = func(df, gap_pct=params["gap_pct"])
                elif name == "Volume Spike":
                    sig = func(df, vol_mult=params["vol_spike_mult"], price_pct=params["vol_price_pct"])
                if sig:
                    sig["ticker"] = rt_ticker
                    rt_signals.append(sig)
            if rt_signals:
                st.dataframe(pd.DataFrame(rt_signals), use_container_width=True)
            else:
                st.info("Сигналов на последнем баре нет.")
    elif not token:
        st.warning("Введите API-токен для получения данных в реальном времени.")
