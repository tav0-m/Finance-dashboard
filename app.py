# ============================================
# Finance Dashboard – Rendimiento, Riesgo y Portafolio Óptimo
# ============================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date

# Optimización
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.objective_functions import L2_reg

# -----------------------
# Configuración página + CSS mínimo
# -----------------------
st.set_page_config(
    page_title="Finance Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: visible;}
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: .2px; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Finance Dashboard: Rendimiento, Riesgo y Optimización de Portafolio")

st.markdown(
    """
    Esta app permite:
    - Descargar precios de **Yahoo Finance**.
    - Calcular **retorno**, **volatilidad** y **Sharpe** por activo.
    - Visualizar **correlaciones** y **drawdowns**.
    - **Optimizar** un portafolio (máx. Sharpe) con restricciones y regularización (**PyPortfolioOpt**).
    """
)

TRADING_DAYS = 252

# -----------------------
# Sidebar (parámetros)
# -----------------------
st.sidebar.header("⚙️ Parámetros")

default_universe = [
    "AAPL","MSFT","AMZN","NVDA","TSLA","META","GOOGL","JPM","XOM","BRK-B"
]

tickers = st.sidebar.multiselect(
    "Elige tickers (máx. 12 para velocidad)",
    options=default_universe,
    default=["AAPL","MSFT","NVDA","AMZN"],
    help="Escribe para buscar. Mantén 4–8 para mejor rendimiento."
)

c1, c2 = st.sidebar.columns(2)
with c1:
    start = st.date_input("Inicio", date(2020,1,1))
with c2:
    end = st.date_input("Término", date.today())

risk_free_annual = st.sidebar.number_input(
    "Tasa libre anual (0.02 = 2%)",
    value=0.0, step=0.005, format="%.3f",
    help="Se usa en Sharpe y en la optimización."
)

st.sidebar.caption("Fuente: Yahoo Finance (Adj Close). Frecuencia: días hábiles.")

if not tickers:
    st.info("Selecciona al menos un ticker en la barra lateral.")
    st.stop()

# -----------------------
# Funciones
# -----------------------
@st.cache_data(show_spinner=False)
def load_prices(_tickers, _start, _end):
    data = yf.download(_tickers, start=_start, end=_end, progress=False, auto_adjust=False)
    prices = data["Adj Close"] if "Adj Close" in data.columns else data
    prices = prices.dropna(how="all").asfreq("B").ffill()
    return prices

@st.cache_data(show_spinner=False)
def compute_core(prices: pd.DataFrame, rf: float):
    returns = prices.pct_change().dropna()
    cumret = (1 + returns).cumprod()

    ann_ret = returns.mean() * TRADING_DAYS
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe  = (ann_ret - rf) / ann_vol

    metrics = pd.DataFrame({
        "Retorno anualizado": ann_ret,
        "Volatilidad anualizada": ann_vol,
        "Sharpe": sharpe
    }).sort_values("Sharpe", ascending=False)

    rolling_max = cumret.cummax()
    drawdown = cumret / rolling_max - 1.0
    max_dd = drawdown.min()

    corr = returns.corr()
    return returns, cumret, metrics, drawdown, max_dd, corr

def plot_cumret(cumret: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10,4))
    for c in cumret.columns:
        ax.plot(cumret.index, cumret[c], label=c)
    ax.set_title("Rentabilidad acumulada (base = 1.0)")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Crecimiento")
    ax.legend(ncol=4, fontsize=8); ax.grid(True, alpha=.3)
    plt.tight_layout()
    return fig

def plot_corr(corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Correlaciones de retornos")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig

def plot_drawdowns(drawdown: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10,3.8))
    for c in drawdown.columns:
        ax.plot(drawdown.index, drawdown[c], label=c)
    ax.set_title("Drawdown (caída desde máximo histórico)")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Drawdown")
    ax.legend(ncol=4, fontsize=8); ax.grid(True, alpha=.3)
    plt.tight_layout()
    return fig

def optimize_portfolio(prices: pd.DataFrame, rf: float, lb: float, ub: float, l2_gamma: float):
    mu = mean_historical_return(prices, frequency=TRADING_DAYS)
    S  = CovarianceShrinkage(prices).ledoit_wolf()
    ef = EfficientFrontier(mu, S, weight_bounds=(lb, ub))
    if l2_gamma and l2_gamma > 0:
        ef.add_objective(L2_reg, gamma=l2_gamma)
    ef.max_sharpe(risk_free_rate=rf)
    weights = pd.Series(ef.clean_weights())
    ret, vol, shp = ef.portfolio_performance(risk_free_rate=rf)
    return weights, ret, vol, shp

# -----------------------
# Carga + cálculos
# -----------------------
with st.spinner("Descargando precios..."):
    prices = load_prices(tickers, start, end)

if prices.empty:
    st.warning("No se descargaron precios. Revisa fechas/tickers.")
    st.stop()

returns, cumret, metrics, drawdown, max_dd, corr = compute_core(prices, risk_free_annual)

st.markdown(
    f"**Rango de datos:** {prices.index.min().date()} → {prices.index.max().date()} · "
    f"**Tickers:** {', '.join(prices.columns)}"
)

# -----------------------
# Tabs
# -----------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📌 Métricas", "📈 Gráficos", "🎯 Optimización", "📦 Datos", "ℹ️ Acerca de"])

with tab1:
    st.subheader("Métricas anualizadas por activo")
    st.dataframe(
        metrics.style.format({
            "Retorno anualizado": "{:.2%}",
            "Volatilidad anualizada": "{:.2%}",
            "Sharpe": "{:.2f}"
        }),
        use_container_width=True
    )

    st.markdown("##### Top 3 por Sharpe")
    cols = st.columns(3)
    top3 = metrics.head(3)
    for i, (name, row) in enumerate(top3.iterrows()):
        with cols[i]:
            st.metric(
                label=name,
                value=f"Sharpe {row['Sharpe']:.2f}",
                delta=f"{row['Retorno anualizado']:.2%} ret · {row['Volatilidad anualizada']:.2%} vol"
            )

with tab2:
    cA, cB = st.columns((2,1))
    with cA:
        st.subheader("Rentabilidad acumulada (base=1)")
        st.pyplot(plot_cumret(cumret), use_container_width=True)
    with cB:
        st.subheader("Correlaciones")
        st.pyplot(plot_corr(corr), use_container_width=True)

    st.subheader("Drawdowns")
    st.pyplot(plot_drawdowns(drawdown), use_container_width=True)

    st.markdown("**Máximos drawdowns:**")
    st.dataframe(
        pd.DataFrame({"Max Drawdown": max_dd}).sort_values("Max Drawdown").style.format({"Max Drawdown": "{:.2%}"}),
        use_container_width=True
    )

with tab3:
    st.subheader("Máx. Sharpe con restricciones y regularización")
    c1, c2, c3 = st.columns(3)
    with c1:
        lb = st.number_input("Límite inferior", 0.0, 1.0, 0.0, 0.05, format="%.2f",
                             help="0 si no permites cortos.")
    with c2:
        ub = st.number_input("Límite superior", 0.05, 1.0, 0.6, 0.05, format="%.2f",
                             help="Evita concentración excesiva.")
    with c3:
        l2 = st.number_input("Regularización L2", 0.0, 0.5, 0.001, 0.001, format="%.3f",
                             help="Estabiliza pesos; mayor = más suavizado.")

    if st.button("Optimizar ahora 🚀"):
        try:
            weights, ret, vol, shp = optimize_portfolio(prices, risk_free_annual, lb, ub, l2)
            st.success(f"Sharpe={shp:.2f} · Ret={ret:.2%} · Vol={vol:.2%}")
            st.dataframe(weights.sort_values(ascending=False).to_frame("peso").style.format("{:.2%}"),
                         use_container_width=True)

            # Backtest buy&hold vs equiponderado
            w = weights.reindex(returns.columns).fillna(0).values
            port_cum = (1 + returns.dot(w)).cumprod()
            eq_cum   = (1 + returns.mean(axis=1)).cumprod()

            fig, ax = plt.subplots(figsize=(10,3.8))
            ax.plot(port_cum.index, port_cum, label="Óptimo")
            ax.plot(eq_cum.index, eq_cum, label="Equiponderado", linestyle="--")
            ax.grid(True, alpha=.3); ax.legend()
            ax.set_title("Backtest (buy & hold)")
            st.pyplot(fig, use_container_width=True)

            st.download_button(
                "Descargar pesos (CSV)",
                weights.to_csv().encode("utf-8"),
                file_name="pesos_optimos.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error en la optimización: {e}")

with tab4:
    c1, c2, c3 = st.columns(3)
    c1.download_button("Precios (CSV)", prices.to_csv().encode("utf-8"), "precios.csv", "text/csv")
    c2.download_button("Retornos (CSV)", returns.to_csv().encode("utf-8"), "retornos.csv", "text/csv")
    c3.download_button("Métricas (CSV)", metrics.to_csv().encode("utf-8"), "metricas.csv", "text/csv")

with tab5:
    st.markdown("""
**Finance Dashboard** — análisis de rendimiento, riesgo y asignación de portafolio.

- Cálculos anualizados con 252 días de mercado.
- Covarianzas **Ledoit–Wolf** y regularización **L2** opcional.
- Sin cortos por defecto (bounds [0,1]).
- Hecho con `Python`, `Streamlit`, `pandas`, `numpy`, `matplotlib`, `yfinance`, `PyPortfolioOpt`.
    """)
