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
    if topic == 'Normální rozdělení':
        st.info("""
        **Co dělají parametry:**
        *   **Střed (μ)**: Posouvá celý zvon doleva nebo doprava po ose x.
        *   **Odchylka (σ)**: Určuje, jak je zvon široký. Malá sigma znamená úzký a vysoký zvon, velká sigma plochý.

        **Kdy se používá:** Pro modelování přirozených jevů, kde se většina hodnot nachází kolem průměru.

        **Reálný příklad:** Výška dospělých lidí v populaci nebo chyby při měření.
        """)
        mu = st.sidebar.slider('Střed (μ):', -5.0, 5.0, 0.0, 0.1)
        sigma = st.sidebar.slider('Odchylka (σ):', 0.1, 5.0, 1.0, 0.1)

        x = np.linspace(-25, 25, 1000)
        y = stats.norm.pdf(x, mu, sigma)
        # Fixní osy zohledňující extrémy parametrů
        _plot_dist(x, y, 'Normální rozdělení', 'royalblue', label=f'μ={mu}, σ={sigma}', xlim=(-25, 25), ylim=(0, 4.0))

    elif topic == 'Rovnoměrné rozdělení':
        st.info("""
        **Co dělají parametry:**
        *   **Minimum (a)** a **Maximum (b)**: Definují hranice, mezi kterými mají všechny hodnoty naprosto stejnou šanci na výskyt.

        **Kdy se používá:** Když víme, že hodnota leží v určitém intervalu, ale nemáme důvod upřednostnit žádnou část tohoto intervalu.

        **Reálný příklad:** Hod kostkou (diskrétní verze) nebo čekání na tramvaj, která jezdí v přesných intervalech.
        """)
        a = st.sidebar.slider('Minimum (a):', 0.0, 5.0, 0.0, 0.1)
        b = st.sidebar.slider('Maximum (b):', 5.1, 10.0, 5.0, 0.1)

        if b <= a: b = a + 0.1
        x = np.linspace(-2, 12, 1000)
        y = stats.uniform.pdf(x, loc=a, scale=b - a)
        _plot_dist(x, y, 'Rovnoměrné rozdělení', 'seagreen', label=f'a={a}, b={b}', xlim=(-2, 12), ylim=(0, 10.5))

    elif topic == 'Exponenciální rozdělení':
        st.info("""
        **Co dělají parametry:**
        *   **Intenzita (λ)**: Vyšší lambda znamená, že události následují rychleji za sebou a graf klesá strměji.

        **Kdy se používá:** Pro modelování času mezi náhodnými událostmi.

        **Reálný příklad:** Čas mezi příchody zákazníků do prodejny nebo životnost elektronické součástky.
        """)
        lam = st.sidebar.slider('Intenzita (λ):', 0.1, 5.0, 1.0, 0.1)
        A = st.sidebar.slider('Start (A):', 0.0, 5.0, 0.0, 0.1)

        x = np.linspace(0, 30, 1000)
        y = stats.expon.pdf(x, loc=A, scale=1 / lam)
        _plot_dist(x, y, 'Exponenciální rozdělení', 'indianred', label=f'λ={lam}, A={A}', xlim=(0, 20), ylim=(0, 5.5))

    elif topic == 'Poissonovo rozdělení':
        st.info("""
        **Co dělají parametry:**
        *   **Intenzita (λ)**: Průměrný počet výskytů události za jednotku času nebo prostoru. Vyšší lambda posouvá těžiště grafu doprava.

        **Kdy se používá:** Pro počítání náhodných událostí v pevném intervalu.

        **Reálný příklad:** Počet e-mailů, které obdržíte za hodinu, nebo počet kazů na metru čtverečním látky.
        """)
        lam = st.sidebar.slider('Lambda (λ):', 0.1, 15.0, 4.0, 0.1)

        k = np.arange(0, 40)
        pmf = stats.poisson.pmf(k, lam)
        fig = plt.figure(figsize=(10, 5))
        plt.bar(k, pmf, color='darkorange', alpha=0.7)
        plt.title('Poissonovo rozdělení')
        plt.xlim(-1, 31)
        plt.ylim(0, 0.5)
        st.pyplot(fig)
        plt.close(fig)

    elif topic == 'Studentovo t-rozdělení':
        st.info("""
        **Co dělají parametry:**
        *   **Stupně volnosti (df)**: Určují tloušťku "chvostů" rozdělení. S rostoucím počtem stupňů volnosti se t-rozdělení blíží normálnímu rozdělení.

        **Kdy se používá:** Při odhadu průměru populace, když máme malý vzorek a neznáme skutečný rozptyl.

        **Reálný příklad:** Testování, zda má nový lék vliv na krevní tlak u malé skupiny pacientů.
        """)
        df = st.sidebar.slider('St. volnosti (df):', 1, 50, 1)

        x = np.linspace(-10, 10, 1000)
        fig = plt.figure(figsize=(10, 5))
        plt.plot(x, stats.t.pdf(x, df), color='royalblue', lw=2, label=f't-dist (df={df})')
        plt.plot(x, stats.norm.pdf(x), color='crimson', linestyle='--', alpha=0.6, label='Normální (referenční)')
        plt.title(f'Studentovo t-rozdělení (df={df})')
        plt.xlim(-5, 5)
        plt.ylim(0, 0.45)
        plt.legend(loc='upper right')
        st.pyplot(fig)
        plt.close(fig)

    elif topic == 'Chi-kvadrát rozdělení':
        st.info("""
        **Co dělají parametry:**
        *   **Stupně volnosti (df)**: Určují tvar rozdělení. Pro malá df je rozdělení silně zešikmené doprava. S rostoucím df se začíná podobat normálnímu rozdělení a posouvá se doprava.

        **Kdy se používá:** Při testování nezávislosti v kontingenčních tabulkách, dobré shody nebo odhadování rozptylu.

        **Reálný příklad:** Testování, zda barva očí souvisí s preferencí politické strany.
        """)
        df = st.sidebar.slider('St. volnosti (df):', 1, 20, 1)

        x = np.linspace(0.1, 40, 1000)
        y = stats.chi2.pdf(x, df)
        _plot_dist(x, y, 'Chi-kvadrát rozdělení', 'darkmagenta', label=f'df={df}', xlim=(0, 40), ylim=(0, 0.5))

    elif topic == 'F-rozdělení':
        st.info("""
        **Co dělají parametry:**
        *   **Stupně volnosti (df1 a df2)**: Určují tvar rozdělení. df1 patří čitateli a df2 jmenovateli. Typicky je rozdělení zešikmené doprava.

        **Kdy se používá:** Primárně pro **testování shody rozptylů** dvou souborů a jako základ pro ANOVA testy.

        **Reálný příklad:** Porovnání přesnosti dvou různých strojů ve výrobě.
        """)
        df1 = st.sidebar.slider('df1:', 1, 100, 5)
        df2 = st.sidebar.slider('df2:', 1, 100, 10)

        x = np.linspace(0.001, 5, 1000)
        y = stats.f.pdf(x, df1, df2)
        _plot_dist(x, y, 'F-rozdělení', 'sienna', label=f'df1={df1}, df2={df2}', xlim=(0, 5), ylim=(0, 2.5))

    elif topic == 'Beta rozdělení':
        st.info("""
        **Co dělají parametry:**
        *   **Alfa (α)** a **Beta (β)**: Tvarové parametry. Pokud jsou obě větší než 1, je rozdělení uprostřed vyklenuté. Pokud je α > β, je hustota nakloněna k 1, při β > α k 0.

        **Kdy se používá:** Pro modelování chování procentuálních podílů nebo pravděpodobností.

        **Reálný příklad:** Odhad procentuální míry prokliku (CTR) u reklamy.
        """)
        a = st.sidebar.slider('Alpha (α):', 0.1, 10.0, 2.0, 0.1)
        b = st.sidebar.slider('Beta (β):', 0.1, 10.0, 5.0, 0.1)

        x = np.linspace(0, 1, 1000)
        y = stats.beta.pdf(x, a, b)
        _plot_dist(x, y, 'Beta rozdělení', 'darkviolet', label=f'α={a}, β={b}', xlim=(0, 1), ylim=(0, 5.0))

    elif topic == 'Log-normální rozdělení':
        st.info("""
        **Co dělají parametry:**
        *   **Mu (μ)** a **Sigma (σ)**: Parametry odpovídající průměru a odchylce logaritmu dané proměnné. Způsobují zešikmení grafu.

        **Kdy se používá:** Pro hodnoty, které nemohou být záporné a mají dlouhý pravý chvost.

        **Reálný příklad:** Rozdělení příjmů v populaci nebo délka hovorů v call centru.
        """)
        mu = st.sidebar.slider('Mu (μ):', -1.0, 2.0, 0.0, 0.1)
        sigma = st.sidebar.slider('Sigma (σ):', 0.1, 1.5, 0.5, 0.1)

        x = np.linspace(0.01, 25, 1000)
        y = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
        _plot_dist(x, y, 'Log-normální rozdělení', 'teal', label=f'μ={mu}, σ={sigma}', xlim=(0, 20), ylim=(0, 1.5))

