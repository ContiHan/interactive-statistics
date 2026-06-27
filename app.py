import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
from statsmodels.tsa.stattools import adfuller

st.set_page_config(page_title="Statistika vizuálně", layout="wide")
plt.style.use('ggplot')

# --- Helpers ---
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
        plt.legend()
    st.pyplot(fig)
    plt.close(fig)

# --- App Layout ---
st.title("Statistika vizuálně")
st.sidebar.header("Navigace")

topics = [
    "Normální rozdělení",
    "Rovnoměrné rozdělení",
    "Exponenciální rozdělení",
    "Poissonovo rozdělení",
    "Studentovo t-rozdělení",
    "Chi-kvadrát rozdělení",
    "F-rozdělení",
    "Beta rozdělení",
    "Log-normální rozdělení",
    "Hustota (PDF) vs Distribuční funkce (CDF)",
    "Centrální limitní věta (CLV)",
    "Chyba I. a II. druhu",
    "ANOVA",
    "Lineární regrese",
    "Analýza reziduí",
    "Dekompozice časové řady",
    "Stacionarita",
    "Q-statistika (Míra diverzity)"
]

selected_topic = st.sidebar.selectbox("Vyberte téma k vizualizaci:", topics)

st.header(selected_topic)

if selected_topic == "Normální rozdělení":
    st.markdown("""
    **Co dělají parametry:**
    *   **Střed (μ)**: Posouvá celý zvon doleva nebo doprava po ose x.
    *   **Odchylka (σ)**: Určuje, jak je zvon široký. Malá sigma znamená úzký a vysoký zvon, velká sigma plochý.
    
    **Kdy se používá:** Pro modelování přirozených jevů, kde se většina hodnot nachází kolem průměru.
    """)
    mu = st.slider('Střed (μ):', -5.0, 5.0, 0.0, 0.1)
    sigma = st.slider('Odchylka (σ):', 0.1, 5.0, 1.0, 0.1)
    
    x = np.linspace(-10, 10, 1000)
    y = stats.norm.pdf(x, mu, sigma)
    _plot_dist(x, y, 'Normální rozdělení', 'royalblue', label=f'μ={mu}, σ={sigma}', xlim=(-10, 10), ylim=(0, 0.9))

elif selected_topic == "Rovnoměrné rozdělení":
    st.markdown("""
    **Co dělají parametry:**
    *   **Minimum (a)** a **Maximum (b)**: Definují hranice, mezi kterými mají všechny hodnoty naprosto stejnou šanci na výskyt.
    """)
    a = st.slider('Minimum (a):', 0.0, 5.0, 0.0, 0.1)
    b = st.slider('Maximum (b):', 5.1, 10.0, 5.0, 0.1)
    
    if b <= a: b = a + 0.1
    x = np.linspace(-2, 12, 1000)
    y = stats.uniform.pdf(x, loc=a, scale=b - a)
    _plot_dist(x, y, 'Rovnoměrné rozdělení', 'seagreen', label=f'a={a}, b={b}', xlim=(-2, 12), ylim=(0, 1.2))

elif selected_topic == "Exponenciální rozdělení":
    st.markdown("""
    **Co dělají parametry:**
    *   **Intenzita (λ)**: Vyšší lambda znamená, že události následují rychleji za sebou a graf klesá strměji.
    """)
    lam = st.slider('Intenzita (λ):', 0.1, 5.0, 1.0, 0.1)
    A = st.slider('Start (A):', 0.0, 5.0, 0.0, 0.1)
    
    x = np.linspace(0, 15, 1000)
    y = stats.expon.pdf(x, loc=A, scale=1 / lam)
    _plot_dist(x, y, 'Exponenciální rozdělení', 'indianred', label=f'λ={lam}, A={A}', xlim=(0, 15), ylim=(0, 5.1))

elif selected_topic == "Poissonovo rozdělení":
    st.markdown("""
    **Co dělají parametry:**
    *   **Intenzita (λ)**: Průměrný počet výskytů události. Vyšší lambda posouvá těžiště grafu doprava.
    """)
    lam = st.slider('Lambda (λ):', 0.1, 15.0, 4.0, 0.1)
    
    k = np.arange(0, 31)
    pmf = stats.poisson.pmf(k, lam)
    fig = plt.figure(figsize=(10, 5))
    plt.bar(k, pmf, color='darkorange', alpha=0.7)
    plt.title('Poissonovo rozdělení')
    plt.ylim(0, 0.5)
    st.pyplot(fig)
    plt.close(fig)

elif selected_topic == "Studentovo t-rozdělení":
    st.markdown("**Stupně volnosti (df)**: Určují tloušťku 'chvostů'. S rostoucím počtem df se t-rozdělení blíží normálnímu.")
    df = st.slider('St. volnosti (df):', 1, 50, 1)
    
    x = np.linspace(-5, 5, 1000)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x, stats.t.pdf(x, df), color='royalblue', label='t-dist')
    plt.plot(x, stats.norm.pdf(x), color='crimson', linestyle='--', label='Normální')
    plt.title(f'Studentovo t-rozdělení (df={df})')
    plt.legend()
    st.pyplot(fig)
    plt.close(fig)

elif selected_topic == "Chi-kvadrát rozdělení":
    st.markdown("**Stupně volnosti (df)**: Pro malá df je rozdělení silně zešikmené. S rostoucím df se podobá normálnímu rozdělení.")
    df = st.slider('St. volnosti (df):', 1, 20, 1)
    
    x = np.linspace(0, 30, 1000)
    y = stats.chi2.pdf(x, df)
    _plot_dist(x, y, 'Chi-kvadrát rozdělení', 'darkmagenta', label=f'df={df}', xlim=(0, 30), ylim=(0, 0.5))

elif selected_topic == "F-rozdělení":
    st.markdown("Primárně pro testování shody rozptylů dvou souborů a jako základ pro ANOVA testy.")
    df1 = st.slider('df1:', 1, 100, 5)
    df2 = st.slider('df2:', 1, 100, 10)
    
    x = np.linspace(0.001, 5, 1000)
    y = stats.f.pdf(x, df1, df2)
    _plot_dist(x, y, 'F-rozdělení', 'sienna', label=f'df1={df1}, df2={df2}', ylim=(0, 2.5))

elif selected_topic == "Beta rozdělení":
    st.markdown("Pro modelování chování procentuálních podílů nebo pravděpodobností.")
    a = st.slider('Alpha (α):', 0.1, 10.0, 2.0, 0.1)
    b = st.slider('Beta (β):', 0.1, 10.0, 5.0, 0.1)
    
    x = np.linspace(0, 1, 1000)
    y = stats.beta.pdf(x, a, b)
    _plot_dist(x, y, 'Beta rozdělení', 'darkviolet', label=f'α={a}, β={b}', xlim=(0, 1), ylim=(0, 5))

elif selected_topic == "Log-normální rozdělení":
    st.markdown("Pro hodnoty, které nemohou být záporné a mají dlouhý pravý chvost.")
    mu = st.slider('Mu (μ):', -1.0, 2.0, 0.0, 0.1)
    sigma = st.slider('Sigma (σ):', 0.1, 1.5, 0.5, 0.1)
    
    x = np.linspace(0.01, 20, 1000)
    y = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    _plot_dist(x, y, 'Log-normální rozdělení', 'teal', label=f'μ={mu}, σ={sigma}', xlim=(0, 20), ylim=(0, 1.5))

elif selected_topic == "Hustota (PDF) vs Distribuční funkce (CDF)":
    dist_type = st.radio("Rozdělení:", ['Normální', 'Exponenciální'])
    
    col1, col2 = st.columns(2)
    with col1:
        if dist_type == 'Normální':
            mu = st.slider('Střed (μ):', -5.0, 5.0, 0.0)
            sigma = st.slider('Odchylka (σ):', 0.5, 3.0, 1.0)
            lam = 1
            x_val = st.slider('Bod x:', mu - 4*sigma, mu + 4*sigma, float(mu), 0.01)
        else:
            mu, sigma = 0, 1
            lam = st.slider('Intenzita (λ):', 0.1, 5.0, 1.0)
            x_val = st.slider('Bod x:', 0.0, 10/lam, 0.0, 0.01)

    if dist_type == 'Normální':
        dist = stats.norm(loc=mu, scale=sigma)
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
        start_x = mu - 4*sigma
    else:
        dist = stats.expon(scale=1/lam)
        x = np.linspace(0, 10/lam, 500)
        start_x = 0
        
    pdf = dist.pdf(x)
    cdf = dist.cdf(x)
    current_cdf = dist.cdf(x_val)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(x, pdf, lw=2, color='royalblue', label='Hustota (PDF)')
    x_fill = np.linspace(start_x, x_val, 100)
    ax1.fill_between(x_fill, dist.pdf(x_fill), alpha=0.3, color='royalblue')
    ax1.axvline(x_val, color='crimson', linestyle='--')
    ax1.set_title(f'Hustota pravděpodobnosti (Plocha = {current_cdf:.3f})')
    ax1.set_xlabel('x')
    ax1.legend()
    
    ax2.plot(x, cdf, lw=2, color='seagreen', label='Distribuční f. (CDF)')
    ax2.axvline(x_val, color='crimson', linestyle='--')
    ax2.axhline(current_cdf, color='crimson', alpha=0.3, linestyle=':')
    ax2.scatter([x_val], [current_cdf], color='crimson')
    ax2.set_title(f'Distribuční funkce (F(x) = {current_cdf:.3f})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('F(x)')
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

elif selected_topic == "Centrální limitní věta (CLV)":
    dist_type = st.radio("Rozdělení:", ['Rovnoměrné', 'Exponenciální'])
    n_samples = st.slider('Velikost (n):', 30, 100, 30, 10)
    n_simulations = st.slider('Počet simulací:', 10, 2000, 100, 10)
    
    np.random.seed(42)
    if dist_type == 'Rovnoměrné':
        samples = np.random.uniform(0, 1, (n_simulations, n_samples))
        mu_theory, sigma_theory = 0.5, np.sqrt(1 / (12 * n_samples))
    else:
        samples = np.random.exponential(scale=1.0, size=(n_simulations, n_samples))
        mu_theory, sigma_theory = 1.0, 1.0 / np.sqrt(n_samples)
        
    means = samples.mean(axis=1)
    dynamic_bins = int(np.clip(np.sqrt(n_simulations) * 1.8 + n_samples / 3, 15, 100))
    fig = plt.figure(figsize=(10, 6))
    _, bins, _ = plt.hist(means, bins=dynamic_bins, density=True, alpha=0.6, color='skyblue', label='Simulované průměry')
    x = np.linspace(min(bins), max(bins), 100)
    plt.plot(x, stats.norm.pdf(x, mu_theory, sigma_theory), 'r', lw=2, label='Teoretické N (CLT)')
    plt.title(f'CLT: Distribuce {n_simulations} průměrů (n={n_samples}, intervalů={dynamic_bins})')
    plt.legend()
    st.pyplot(fig)
    plt.close(fig)

elif selected_topic == "Chyba I. a II. druhu":
    mu_alt = st.slider('Efekt (μ1):', 0.5, 5.0, 2.5, 0.1)
    alpha = st.slider('Alfa (α):', 0.01, 0.20, 0.05, 0.01)
    sigma = st.slider('Šum (σ):', 0.5, 2.0, 1.0, 0.1)
    
    mu0 = 0
    critical_val = stats.norm.ppf(1 - alpha, mu0, sigma)
    x = np.linspace(-4, mu_alt + 4, 1000)
    y0 = stats.norm.pdf(x, mu0, sigma)
    y1 = stats.norm.pdf(x, mu_alt, sigma)
    beta = stats.norm.cdf(critical_val, mu_alt, sigma)
    mask_alpha = x >= critical_val
    mask_beta = x <= critical_val
    
    fig = plt.figure(figsize=(12, 6))
    plt.plot(x, y0, label='H0 (Nulová)', color='royalblue', lw=2)
    plt.plot(x, y1, label='H1 (Alternativní)', color='crimson', lw=2)
    plt.fill_between(x[mask_alpha], y0[mask_alpha], color='royalblue', alpha=0.4, label=f'Alfa (I. druh) = {alpha:.3f}')
    plt.fill_between(x[mask_beta], y1[mask_beta], color='crimson', alpha=0.4, label=f'Beta (II. druh) = {beta:.3f}')
    plt.axvline(critical_val, color='black', linestyle='--', label='Kritická mez')
    plt.title(f'Testování hypotéz: Síla testu (1-β) = {1-beta:.3f}')
    plt.legend()
    st.pyplot(fig)
    plt.close(fig)

elif selected_topic == "ANOVA":
    mu1 = st.slider('μ Skupina 1:', 0.0, 10.0, 4.0, 0.5)
    mu2 = st.slider('μ Skupina 2:', 0.0, 10.0, 5.0, 0.5)
    mu3 = st.slider('μ Skupina 3:', 0.0, 10.0, 6.0, 0.5)
    sigma = st.slider('Společné σ:', 1.0, 20.0, 1.0, 1.0)
    
    x = np.linspace(min(mu1, mu2, mu3) - 4*sigma, max(mu1, mu2, mu3) + 4*sigma, 1000)
    means = np.array([mu1, mu2, mu3])

    np.random.seed(42)
    groups = [np.random.normal(mu, sigma, 200) for mu in means]
    f_stat, p_val = stats.f_oneway(*groups)

    fig = plt.figure(figsize=(12, 6))
    colors = ['royalblue', 'seagreen', 'indianred']
    for i, (mu, color) in enumerate(zip(means, colors)):
        y = stats.norm.pdf(x, mu, sigma)
        plt.plot(x, y, label=f'Skupina {i+1} (μ={mu})', color=color, lw=2)
        plt.fill_between(x, y, alpha=0.1, color=color)
        plt.axvline(mu, color=color, linestyle='--', alpha=0.5)

    mean_of_means = np.mean(means)
    std_of_means = np.std(means, ddof=1) if len(means) > 1 else 0.1
    x_means = np.linspace(mean_of_means - 4*std_of_means, mean_of_means + 4*std_of_means, 100)
    y_means = stats.norm.pdf(x_means, mean_of_means, std_of_means)

    plt.plot(x_means, y_means, color='black', lw=3, label='Rozdělení průměrů (Meziskupinové)')
    plt.fill_between(x_means, y_means, alpha=0.2, color='black', hatch='//')

    sig_text = 'p < 0.05 → Signifikantní' if p_val < 0.05 else 'p ≥ 0.05 → Nesignifikantní'
    title_color = 'darkred' if p_val < 0.05 else 'steelblue'
    plt.title(f'ANOVA: F = {f_stat:.2f}, p = {p_val:.4f} | {sig_text}', color=title_color)
    plt.legend()
    st.pyplot(fig)
    plt.close(fig)

elif selected_topic == "Lineární regrese":
    slope = st.slider('Sklon:', -2.0, 2.0, 1.5, 0.1)
    intercept = st.slider('Průsečík:', -5.0, 5.0, 2.0, 0.1)
    noise = st.slider('Šum:', 0.1, 10.0, 2.0, 0.1)
    
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = slope * x + intercept + np.random.normal(0, noise, 50)
    res = stats.linregress(x, y)
    fig = plt.figure(figsize=(10, 5))
    plt.scatter(x, y, alpha=0.6, color='slateblue')
    plt.plot(x, res.slope * x + res.intercept, color='crimson', label=f'R² = {res.rvalue**2:.3f}')
    plt.title('Lineární regrese')
    plt.xlim(0, 10)
    plt.ylim(-10, 30)
    plt.legend()
    st.pyplot(fig)
    plt.close(fig)

elif selected_topic == "Analýza reziduí":
    mean_error = st.slider('Střed chyby:', -5.0, 5.0, 0.0, 0.5)
    scale_error = st.slider('Zákl. rozptyl:', 0.5, 10.0, 2.0, 0.5)
    heteroskedasticity = st.slider('Trychtýř:', 0.0, 10.0, 0.0, 0.5)
    normality = st.checkbox('Normalita', value=True)
    
    np.random.seed(42)
    x = np.linspace(0, 100, 200)
    if normality:
        errors = np.random.normal(loc=mean_error, scale=scale_error, size=200)
    else:
        errors = np.random.exponential(scale=scale_error, size=200) - scale_error
    if heteroskedasticity > 0:
        noise_growth = (x / 100) * heteroskedasticity
        errors = errors * (1 + noise_growth)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.scatter(x, errors, alpha=0.6, color='darkorchid')
    ax1.axhline(0, color='black', linestyle='--', lw=1)
    ax1.set_title('Graf reziduí')
    ax1.set_ylim(-25, 25)
    
    sns.histplot(errors, kde=True, ax=ax2, color='darkorchid', alpha=0.4)
    ax2.set_title('Rozdělení reziduí')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

elif selected_topic == "Dekompozice časové řady":
    trend_slope = st.slider('Trend:', -3.0, 3.0, 0.5, 0.05)
    seasonal_amp = st.slider('Sezónnost:', 0, 50, 10)
    cycle_amp = st.slider('Cyklus:', 0, 50, 15)
    noise_level = st.slider('Šum:', 0.0, 20.0, 2.0, 0.5)
    
    t = np.arange(120)
    dates = pd.date_range(start='2015-01-01', periods=120, freq='MS')
    trend = trend_slope * t
    seasonality = seasonal_amp * np.sin(2 * np.pi * t / 12)
    cycle = cycle_amp * np.sin(2 * np.pi * t / (12 * 4))
    np.random.seed(42)
    noise = np.random.normal(0, noise_level, len(t))
    total = trend + seasonality + cycle + noise
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 8), sharex=True)
    components = [
        (total, 'Výsledná řada', 'black'), (trend, 'Trend', 'royalblue'),
        (seasonality, 'Sezónnost', 'teal'), (cycle, 'Cyklus', 'goldenrod'),
        (noise, 'Rezidua', 'firebrick')
    ]
    for i, (ax, (data, title, color)) in enumerate(zip(axes, components)):
        ax.plot(dates, data, color=color, lw=1.5 if i==0 else 1.2)
        ax.set_title(title, fontsize=10, pad=5)
        ax.grid(True, alpha=0.15)
    plt.tight_layout(h_pad=0.4)
    st.pyplot(fig)
    plt.close(fig)

elif selected_topic == "Stacionarita":
    phi = st.slider('Koeficient (φ):', 0.0, 1.03, 0.5, 0.01)
    noise_std = st.slider('Šum:', 0.1, 5.0, 1.0, 0.1)
    
    np.random.seed(42)
    n = 200
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t-1] + np.random.normal(0, noise_std)
    res = adfuller(y)
    p_val = res[1]
    
    fig = plt.figure(figsize=(12, 5))
    plt.plot(y, color='darkblue', lw=1.5)
    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    plt.ylim(-30, 30)
    status = "STACIONÁRNÍ" if phi < 1 else "NESTACIONÁRNÍ"
    plt.title(f"AR(1) s φ={phi} | p-hodnota: {p_val:.4f} | {status}")
    st.pyplot(fig)
    plt.close(fig)

elif selected_topic == "Q-statistika (Míra diverzity)":
    n_common_correct = st.slider('Oba OK (a):', 0, 100, 60)
    n_only_b_wrong = st.slider('A OK, B KO (b):', 0, 50, 10)
    n_only_a_wrong = st.slider('A KO, B OK (c):', 0, 50, 10)
    n_both_wrong = st.slider('Oba KO (d):', 0, 100, 20)
    
    a, b, c, d = n_common_correct, n_only_b_wrong, n_only_a_wrong, n_both_wrong
    denominator = a * d + b * c
    q_val = (a * d - b * c) / denominator if denominator != 0 else 0
    data = [[a, b], [c, d]]
    
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(data, annot=True, fmt='d', cmap='RdYlGn_r',
                xticklabels=['B Správně', 'B Chyba'], yticklabels=['A Správně', 'A Chyba'])
    plt.title(f'Matice chyb\nQ-statistika = {q_val:.3f}')
    st.pyplot(fig)
    plt.close(fig)
