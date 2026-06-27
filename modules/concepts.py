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
    if topic == 'Hustota (PDF) vs Distribuční funkce (CDF)':
        st.info("""
        **Co uvidíte v grafech:**
        *   **Hustota (PDF - vlevo)**: Ukazuje šanci na výskyt konkrétní hodnoty. Vybarvená plocha pod křivkou od -∞ do bodu x představuje celkovou pravděpodobnost.
        *   **Distribuční f. (CDF - vpravo)**: Ukazuje kumulativní pravděpodobnost. Hodnota na ose Y v bodě x přesně odpovídá velikosti vybarvené plochy v levém grafu.

        **Kdy se to hodí:** Když potřebujete vědět, jaká je šance, že výsledek bude menší nebo roven určité hodnotě.
        """)
        dist_type = st.sidebar.radio("Rozdělení:", ['Normální', 'Exponenciální'])

        if dist_type == 'Normální':
            mu = st.sidebar.slider('Střed (μ):', -5.0, 5.0, 0.0)
            sigma = st.sidebar.slider('Odchylka (σ):', 0.5, 3.0, 1.0)
            lam = 1
            x_val = st.sidebar.slider('Bod x (kde chci zjistit plochu):', -15.0, 15.0, float(mu), 0.1)
        else:
            mu, sigma = 0, 1
            lam = st.sidebar.slider('Intenzita (λ):', 0.1, 5.0, 1.0)
            x_val = st.sidebar.slider('Bod x (kde chci zjistit plochu):', 0.0, 50.0, 1.0, 0.1)

        if dist_type == 'Normální':
            dist = stats.norm(loc=mu, scale=sigma)
            x = np.linspace(-20, 20, 500)
            start_x = -20
            x_limit, y_limit = (-15, 15), (0, 1.0)
        else:
            dist = stats.expon(scale=1/lam)
            x = np.linspace(0, 60, 500)
            start_x = 0
            x_limit, y_limit = (0, 40), (0, 5.5)

        pdf = dist.pdf(x)
        cdf = dist.cdf(x)
        current_cdf = dist.cdf(x_val)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(x, pdf, lw=2, color='royalblue', label='Hustota (PDF)')
        # Fix x_val visualization limits so fill area doesn't break if x_val is out of theoretical bounds
        x_val_capped = min(max(x_val, start_x), x[-1])
        x_fill = np.linspace(start_x, x_val_capped, 100)
        ax1.fill_between(x_fill, dist.pdf(x_fill), alpha=0.3, color='royalblue')
        ax1.axvline(x_val, color='crimson', linestyle='--')
        ax1.set_title(f'Hustota (Plocha = {current_cdf:.3f})')
        ax1.set_xlabel('x')
        ax1.set_xlim(*x_limit)
        ax1.set_ylim(*y_limit)
        ax1.legend()

        ax2.plot(x, cdf, lw=2, color='seagreen', label='Distribuční f. (CDF)')
        ax2.axvline(x_val, color='crimson', linestyle='--')
        ax2.axhline(current_cdf, color='crimson', alpha=0.3, linestyle=':')
        ax2.scatter([x_val], [current_cdf], color='crimson')
        ax2.set_title(f'Distribuční funkce (F(x) = {current_cdf:.3f})')
        ax2.set_xlabel('x')
        ax2.set_ylabel('F(x)')
        ax2.set_xlim(*x_limit)
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    elif topic == 'Centrální limitní věta (CLV)':
        st.info("""
        **Jak se za jakéhokoli rozdělení stane normální rozdělení (zvon)**:
        *   **Velikost (n)**: Počet prvků pro výpočet jednoho průměru (typicky potřebujeme n ≥ 30).
        *   **Počet simulací**: Kolikrát provedeme výběr a spočítáme průměr. Čím více, tím jasnější zvon.
        """)
        dist_type = st.sidebar.radio("Základní rozdělení:", ['Rovnoměrné (Kostky)', 'Exponenciální (Čekání)'])
        n_samples = st.sidebar.slider('Velikost výběru (n):', 30, 100, 30, 10)
        n_simulations = st.sidebar.slider('Počet simulací:', 10, 2000, 100, 10)

        np.random.seed(42)
        if 'Rovnoměrné' in dist_type:
            samples = np.random.uniform(0, 1, (n_simulations, n_samples))
            mu_theory, sigma_theory = 0.5, np.sqrt(1 / (12 * n_samples))
            x_limit, y_limit = (0.2, 0.8), (0, 12)
        else:
            samples = np.random.exponential(scale=1.0, size=(n_simulations, n_samples))
            mu_theory, sigma_theory = 1.0, 1.0 / np.sqrt(n_samples)
            x_limit, y_limit = (0.4, 1.8), (0, 4.0)

        means = samples.mean(axis=1)
        dynamic_bins = int(np.clip(np.sqrt(n_simulations) * 1.8 + n_samples / 3, 15, 100))

        fig = plt.figure(figsize=(10, 6))
        plt.hist(means, bins=dynamic_bins, density=True, alpha=0.6, color='skyblue', label='Simulované průměry')
        x = np.linspace(x_limit[0]-1, x_limit[1]+1, 200)
        plt.plot(x, stats.norm.pdf(x, mu_theory, sigma_theory), 'r', lw=2, label='Teoretické N (CLT)')
        plt.title(f'CLT: Distribuce {n_simulations} průměrů (n={n_samples})')
        plt.xlim(*x_limit)
        plt.ylim(*y_limit)
        plt.legend(loc='upper right')
        st.pyplot(fig)
        plt.close(fig)

    elif topic == 'Zákon velkých čísel (LLN)':
        st.info("""
        **Zákon velkých čísel (Law of Large Numbers)**
        Čím více náhodných pokusů uděláme, tím více se jejich průměr blíží očekávané teoretické hodnotě.

        *   **Příklad ze života (Hody kostkou):** Když hodíte kostkou desetkrát, průměr může být klidně 5 (protože vám padaly samé šestky). Na začátku grafu tak čára divoce skáče nahoru a dolů. Ale pokud hodíte 1000krát, průměr se nevyhnutelně usadí na teoretických 3.5 (červená čára). Kasina na tomto zákonu staví celý svůj byznys.
        """)
        n_flips = st.sidebar.slider('Počet hodů (N):', 10, 2000, 500, 10)

        np.random.seed(42)
        rolls = np.random.randint(1, 7, n_flips)
        cumulative_avg = np.cumsum(rolls) / np.arange(1, n_flips + 1)

        fig = plt.figure(figsize=(12, 6))
        plt.plot(range(1, n_flips + 1), cumulative_avg, color='royalblue', lw=2, alpha=0.8, label='Průběžný průměr hodů')
        plt.axhline(3.5, color='crimson', linestyle='--', lw=2, label='Teoretický průměr (3.5)')

        plt.title('Zákon velkých čísel (Hody kostkou)')
        plt.xlabel('Počet hodů')
        plt.ylabel('Průměrná hodnota')
        plt.xlim(0, 2010)
        plt.ylim(1.5, 5.5)
        plt.legend(loc='upper right')
        st.pyplot(fig)
        plt.close(fig)

