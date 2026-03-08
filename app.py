from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List

import pandas as pd
import streamlit as st

try:
    import akshare as ak
except Exception:  # pragma: no cover
    ak = None


@dataclass
class StrategyConfig:
    lookback_days: int = 160
    ma_fast: int = 20
    ma_slow: int = 60
    high_window: int = 120
    min_price_above_mid: float = 0.52
    ma_slope_window: int = 10


def is_main_board(code: str) -> bool:
    code = str(code)
    return code.startswith(("600", "601", "603", "605", "000", "001"))


@st.cache_data(ttl=60 * 10)
def get_spot_main_board() -> pd.DataFrame:
    if ak is None:
        raise RuntimeError("未安装 akshare，请先执行 `pip install -r requirements.txt`。")

    spot = ak.stock_zh_a_spot_em()
    spot["代码"] = spot["代码"].astype(str).str.zfill(6)
    spot = spot[spot["代码"].map(is_main_board)].copy()
    keep_cols = [
        "代码",
        "名称",
        "最新价",
        "涨跌幅",
        "成交量",
        "成交额",
        "换手率",
    ]
    return spot[keep_cols]


@st.cache_data(ttl=60 * 30)
def get_hist(symbol: str, days: int) -> pd.DataFrame:
    if ak is None:
        raise RuntimeError("未安装 akshare，请先执行 `pip install -r requirements.txt`。")

    start = (date.today() - timedelta(days=days * 2)).strftime("%Y%m%d")
    end = date.today().strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start,
        end_date=end,
        adjust="qfq",
    )
    if df.empty:
        return df

    df = df.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.tail(days).reset_index(drop=True)


def pick_right_side_risers(codes: List[str], cfg: StrategyConfig) -> pd.DataFrame:
    rows = []
    for code in codes:
        try:
            hist = get_hist(code, cfg.lookback_days)
        except Exception:
            continue

        if len(hist) < max(cfg.ma_slow, cfg.high_window) + cfg.ma_slope_window:
            continue

        hist["ma_fast"] = hist["close"].rolling(cfg.ma_fast).mean()
        hist["ma_slow"] = hist["close"].rolling(cfg.ma_slow).mean()

        latest = hist.iloc[-1]
        prev_slow = hist["ma_slow"].iloc[-cfg.ma_slope_window]
        if pd.isna(prev_slow) or pd.isna(latest["ma_fast"]) or pd.isna(latest["ma_slow"]):
            continue

        swing_high = hist["high"].iloc[-cfg.high_window:].max()
        swing_low = hist["low"].iloc[-cfg.high_window:].min()
        range_pos = (latest["close"] - swing_low) / max(swing_high - swing_low, 1e-9)

        cond_trend = latest["close"] > latest["ma_fast"] > latest["ma_slow"]
        cond_slope = latest["ma_slow"] > prev_slow
        cond_right_side = cfg.min_price_above_mid <= range_pos < 0.98

        if cond_trend and cond_slope and cond_right_side:
            rows.append(
                {
                    "代码": code,
                    "最新收盘": round(float(latest["close"]), 2),
                    "MA20": round(float(latest["ma_fast"]), 2),
                    "MA60": round(float(latest["ma_slow"]), 2),
                    "120日区间位置": round(float(range_pos), 3),
                    "120日高点": round(float(swing_high), 2),
                }
            )

    return pd.DataFrame(rows).sort_values(by="120日区间位置", ascending=False)


def run() -> None:
    st.set_page_config(page_title="A股主板右侧上升筛选", layout="wide")
    st.title("A股主板右侧上升区间筛选")
    st.caption("逻辑：收盘价 > MA20 > MA60，MA60上拐，且价格位于近120日振幅的右侧区间（默认52%~98%）。")

    c1, c2, c3 = st.columns(3)
    lookback_days = c1.slider("回看天数", 120, 260, 160, 5)
    min_pos = c2.slider("右侧区间下限", 0.45, 0.80, 0.52, 0.01)
    auto_refresh = c3.checkbox("每60秒自动刷新", value=False)

    if st.button("立即刷新"):
        st.cache_data.clear()

    if auto_refresh:
        st.markdown(
            "<meta http-equiv='refresh' content='60'>",
            unsafe_allow_html=True,
        )

    try:
        spot = get_spot_main_board()
    except Exception as exc:
        st.error(f"获取行情失败：{exc}")
        return

    st.info(f"当前主板股票数量：{len(spot)}")

    cfg = StrategyConfig(lookback_days=lookback_days, min_price_above_mid=min_pos)
    result = pick_right_side_risers(spot["代码"].tolist(), cfg)

    if result.empty:
        st.warning("当前无满足条件的主板股票。可以降低右侧区间下限后重试。")
        return

    merged = result.merge(spot, on="代码", how="left")
    show_cols = ["代码", "名称", "最新价", "涨跌幅", "换手率", "最新收盘", "MA20", "MA60", "120日区间位置", "120日高点"]
    st.dataframe(merged[show_cols], use_container_width=True, hide_index=True)

    pick = st.selectbox("查看个股近期走势", merged["代码"] + " " + merged["名称"])
    pick_code = pick.split()[0]
    hist = get_hist(pick_code, lookback_days)
    hist["MA20"] = hist["close"].rolling(20).mean()
    hist["MA60"] = hist["close"].rolling(60).mean()
    st.line_chart(hist.set_index("date")[["close", "MA20", "MA60"]])


if __name__ == "__main__":
    run()
