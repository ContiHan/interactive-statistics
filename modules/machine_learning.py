import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs
import matplotlib.patches as mpatches

plt.style.use('ggplot')

def render(topic):
    if topic == 'Rozhodovací stromy':
        st.info("""
        **Rozhodovací stromy (Strojové učení / Klasifikace)**
        Tento algoritmus se snaží najít jednoduchá pravidla (čáry), která nejlépe oddělí různé skupiny dat.
        
        *   **Příklad ze života:** Zákazníci banky (osa X = Věk, osa Y = Příjem). Modří si půjčku nevezmou, Červení ano. Chceme najít to nejlepší pravidlo, např. "Pokud je věk > 40 a příjem < 50k, pak..."
        *   **Jak to funguje:** Stroj iterativně seká prostor horizontálně a vertikálně. Cílem je, aby každý vzniklý obdélník obsahoval pokud možno body jedné barvy (maximalizace čistoty / minimalizace entropie).
        *   **Tip:** Vyzkoušejte změnit **Hloubku stromu**. Hloubka 1 udělá jen jeden řez, což často nestačí. Hloubka 10 rozřeže prostor tak moc, že vznikne tzv. "Přeučení" (Overfitting), kdy se strom učí nazpaměť konkrétní body, místo aby chápal obecný vzor.
        """)
        
        depth = st.sidebar.slider('Maximální hloubka stromu:', 1, 10, 2, 1)
        cluster_std = st.sidebar.slider('Rozptyl bodů (Překryv skupin):', 0.5, 3.0, 1.5, 0.1)
        
        np.random.seed(42)
        X, y = make_blobs(n_samples=300, centers=[[3, 3], [7, 7]], cluster_std=cluster_std, random_state=42)
        
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X, y)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                             np.arange(y_min, y_max, 0.05))
        
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=40)
        
        ax.set_title(f'Rozhodovací strom (Hloubka: {depth})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Věk (relativní metrika X)')
        ax.set_ylabel('Příjem (relativní metrika Y)')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        patch0 = mpatches.Patch(color='#7a93ce', label='Skupina 0 (Modrá: Ne)')
        patch1 = mpatches.Patch(color='#d67972', label='Skupina 1 (Červená: Ano)')
        ax.legend(handles=[patch0, patch1], loc='lower right')
        
        st.pyplot(fig)
        plt.close(fig)
