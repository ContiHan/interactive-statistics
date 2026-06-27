import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LogisticRegression

plt.style.use('ggplot')

def _plot_dist(x, y, title, color, label=None, xlim=None, ylim=None, figsize=(10, 5)):
    fig = plt.figure(figsize=figsize)
    plt.plot(x, y, lw=2, color=color, label=label)
    plt.fill_between(x, y, alpha=0.2, color=color)
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Hustota pravděpodobnosti')
    if label is not None:
        plt.legend(loc='upper right')
    st.pyplot(fig)
    plt.close(fig)

def render(topic):
    if topic == 'Dekompozice časové řady':
        st.info("""
        **Složky časové řady:**
        *   **Trend (Sklon)**: Dlouhodobý směr dat (růst nebo pokles).
        *   **Sezónnost (Amplituda)**: Krátkodobé, pravidelně se opakující výkyvy (např. vánoční nákupy - trvají 1 rok).
        *   **Cyklus (Amplituda)**: Dlouhodobé vlny (např. hospodářský cyklus).
        *   **Rezidua (Šum)**: Náhodné a nepředvídatelné kolísání.
        """)
        trend_slope = st.sidebar.slider('Trend:', -3.0, 3.0, 0.5, 0.05)
        seasonal_amp = st.sidebar.slider('Sezónnost (roční):', 0, 50, 10)
        cycle_amp = st.sidebar.slider('Cyklus (víceletý):', 0, 50, 15)
        noise_level = st.sidebar.slider('Šum:', 0.0, 20.0, 2.0, 0.5)

        t = np.arange(120)
        dates = pd.date_range(start='2015-01-01', periods=120, freq='MS')
        trend = trend_slope * t
        seasonality = seasonal_amp * np.sin(2 * np.pi * t / 12)
        cycle = cycle_amp * np.sin(2 * np.pi * t / (12 * 4))
        np.random.seed(42)
        noise = np.random.normal(0, noise_level, len(t))
        total = trend + seasonality + cycle + noise

        fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
        components = [
            (total, 'Výsledná řada (Součet všech níže)', 'black', (-400, 500)), 
            (trend, 'Trend', 'royalblue', (-400, 400)),
            (seasonality, 'Sezónnost', 'teal', (-60, 60)), 
            (cycle, 'Cyklus', 'goldenrod', (-60, 60)),
            (noise, 'Rezidua (Šum)', 'firebrick', (-60, 60))
        ]
        for i, (ax, (data, title, color, y_lim)) in enumerate(zip(axes, components)):
            ax.plot(dates, data, color=color, lw=1.5 if i==0 else 1.2)
            ax.set_title(title, fontsize=10, pad=5)
            ax.grid(True, alpha=0.15)
            ax.set_ylim(*y_lim)
        plt.tight_layout(h_pad=0.4)
        st.pyplot(fig)
        plt.close(fig)

    elif topic == 'Stacionarita':
        st.info("""
        **Stacionarita (pravidla hry se nemění)** znamená, že časová řada má konstantní průměr, rozptyl a chová se stabilně.
        *   **Stacionární (φ < 1)**: Jako Bungee jumper. Vždycky se vrátí k průměru.
        *   **Jednotkový kořen (φ = 1)**: Jako Opilec na louce (Random walk). Kam šlápne, tam stojí, průměr nedává smysl.
        *   **Explozivní (φ > 1)**: Raketa, čísla okamžitě letí do extrémů.
        """)
        phi = st.sidebar.slider('Koeficient paměti (φ):', 0.0, 1.03, 0.5, 0.01)
        noise_std = st.sidebar.slider('Šum:', 0.1, 5.0, 1.0, 0.1)

        np.random.seed(42)
        n = 200
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t-1] + np.random.normal(0, noise_std)

        try:
            res = adfuller(y)
            p_val = res[1]
        except Exception:
            p_val = 1.0 # Fallback for explosive series where ADF fails

        fig = plt.figure(figsize=(12, 5))
        plt.plot(y, color='darkblue', lw=1.5)
        plt.axhline(0, color='black', alpha=0.3, linestyle='--')

        # Pro stacionaritu a random walk udržíme fixní okno, pro explozi ať to raději uletí
        plt.ylim(-50, 50)
        plt.xlim(0, 200)

        status = "STACIONÁRNÍ" if phi < 1 else "NESTACIONÁRNÍ (Random Walk / Explozivní)"
        plt.title(f"Autoregresní proces AR(1) s φ={phi} | p-hodnota (ADF): {p_val:.4f}\n{status}")
        st.pyplot(fig)
        plt.close(fig)

    elif topic == 'Q-statistika (Míra diverzity)':
        st.info("""
        Q-statistika měří, jak moc jsou chyby dvou modelů závislé (pro Ensemble strojové učení).
        *   **Q ≈ 0**: Modely se pletou nezávisle na sobě (ideál).
        *   **Q > 0**: Kladná korelace. Když se splete A, splete se často i B (špatně).
        *   **Q < 0**: Záporná korelace. Modely se doplňují.
        """)
        n_common_correct = st.sidebar.slider('Oba správně (a):', 0, 100, 60)
        n_only_b_wrong = st.sidebar.slider('A správně, B chyba (b):', 0, 50, 10)
        n_only_a_wrong = st.sidebar.slider('A chyba, B správně (c):', 0, 50, 10)
        n_both_wrong = st.sidebar.slider('Oba chyba (d):', 0, 100, 20)

        a, b, c, d = n_common_correct, n_only_b_wrong, n_only_a_wrong, n_both_wrong
        denominator = a * d + b * c
        q_val = (a * d - b * c) / denominator if denominator != 0 else 0
        data = [[a, b], [c, d]]

        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(data, annot=True, fmt='d', cmap='RdYlGn_r',
                    xticklabels=['B Správně', 'B Chyba'], yticklabels=['A Správně', 'A Chyba'], vmin=0, vmax=100)
        plt.title(f'Matice chyb\nQ-statistika = {q_val:.3f}')
        st.pyplot(fig)
        plt.close(fig)

