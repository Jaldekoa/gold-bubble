import streamlit as st
from lppls import lppls_cmaes, lppls_lm
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime as dt, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="Gold Bubble Analyzer", page_icon="🎈", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
            h1 {text-align: center; padding-bottom: 1rem !important;}
            .stHorizontalBlock {padding: 1rem !important;}            
    </style>
""", unsafe_allow_html=True)

st.title("🎈 The Gold Bubble (GC=F) 🎈")

ticker = "GC=F"

DATA_START = "2022-01-01"

@st.cache_data(ttl="1d")
def get_data(ticker, start):
    end = dt.today().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start, end=end, multi_level_index=False, auto_adjust=False)
    df.reset_index(inplace=True)
    return df

@st.cache_data(ttl="1d")
def find_bubble_start(df):
    # Preparación de datos
    time = np.arange(len(df))
    time_scaled = (time - time.min()) / (time.max() - time.min())
    price = np.log(df['Adj Close'].values)
    price_scaled = (price - price.min()) / (price.max() - price.min())
    
    observations = np.array([time_scaled, price_scaled])
    lppls_model = lppls_lm.LPPLS(observations)
    
    # Detección
    result = lppls_model.detect_bubble_start_time_via_lagrange(
        max_window_size=len(time),
        min_window_size=100,
        step_size=3,
        max_searches=25
    )
    
    if result:
        # Convertir tau (escala 0-1) a índice real y luego a fecha
        tau_idx = int(result['tau'] * (len(time) - 1))
        # Asegurarnos de no salirnos del índice
        tau_idx = max(0, min(tau_idx, len(df)-1))
        start_date_detected = df['Date'].iloc[tau_idx]
        return start_date_detected
    return None

@st.cache_data(ttl="1d")
def compute_lppls_parameters(time, price_log):
    observations = np.array([time, price_log])
    lppls_model = lppls_cmaes.LPPLSCMAES(observations=observations)
    params = lppls_model.fit(max_iteration=5000, pop_size=8)
    return params

# --- FLUJO DE EJECUCIÓN ---
full_data = get_data(ticker, DATA_START)

if not full_data.empty:
    with st.spinner('Detection of bubble formation...'):
        optimal_start_date = find_bubble_start(full_data)
    
    if optimal_start_date:
        # Filtrar datos desde el inicio detectado para el ajuste final
        data = full_data[full_data['Date'] >= optimal_start_date].copy()
        
        time_ord = [date.toordinal() for date in data['Date']]
        price_log = np.log(data['Adj Close'].values)
        
        with st.spinner('Calculating bubble fit...'):
            res = compute_lppls_parameters(time_ord, price_log)
            tc, m, w, a, b, c, c1, c2, O, D = res

        tc_date = dt.fromordinal(int(tc))
        
        # --- CÁLCULO DE CURVAS ---
        # El gráfico muestra desde el Optimal Start hasta tc + 45
        end_visual_ordinal = int(tc) + 45
        time_extended_visual = np.array(range(time_ord[0], end_visual_ordinal))
        
        # Curva truncada en tc
        time_lppls_fit = np.array(range(time_ord[0], int(tc) + 1))
        model_fast = lppls_cmaes.LPPLSCMAES(observations=np.array([time_ord, price_log]))
        fit_log_truncated = model_fast.lppls(time_lppls_fit, tc, m, w, a, b, c1, c2)
        fit_nominal_truncated = np.exp(fit_log_truncated)
        
        price_at_tc = fit_nominal_truncated[-1]

        # --- MÉTRICAS SUPERIORES ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Bubble Start", optimal_start_date.strftime('%Y-%m-%d'))
        col2.metric("Bursting Date", tc_date.strftime('%Y-%m-%d'))
        col3.metric("Bursting Price", f"${price_at_tc:,.2f}")

        # --- GRÁFICOS ---
        tab_log, tab_nominal = st.tabs(["📉 Logarithmic", "💵 Nominal (USD)"])

        def create_fig(y_real, y_pred, label):
            fig = go.Figure()
            dates_visual = [dt.fromordinal(int(o)) for o in time_extended_visual]
            dates_fit = [dt.fromordinal(int(o)) for o in time_lppls_fit]
            
            # Histórico Real (mostramos solo desde el optimal start para mayor claridad)
            fig.add_trace(go.Scatter(x=data['Date'], y=y_real, name="Real Price", line=dict(color="gold", width=1.5)))
            
            # Ajuste LPPLS Truncado
            fig.add_trace(go.Scatter(x=dates_fit, y=y_pred, name="LPPLS Fit", line=dict(color="cyan", dash="dot", width=2)))
            
            # Línea tc vertical
            fig.add_vline(x=tc_date, line_width=2, line_dash="dash", line_color="red")
            
            fig.update_layout(
                template="plotly_dark", height=450, hovermode="x unified",
                xaxis=dict(range=[dates_visual[0], dates_visual[-1]]),
                xaxis_title=None, yaxis_title=label,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, orientation="h"),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            return fig

        with tab_log:
            st.plotly_chart(create_fig(price_log, fit_log_truncated, "Log(Price)"), width='stretch', config={'displayModeBar': False})

        with tab_nominal:
            st.plotly_chart(create_fig(data['Adj Close'].values, fit_nominal_truncated, "Price (USD)"), width='stretch', config={'displayModeBar': False})
    else:
        st.error("Could not determine a bubble start time.")
else:
    st.error("No data could be loaded.")