import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.style.use('ggplot')

def render(topic):
    if topic == 'Bayesův teorém':
        st.info("""
        **Bayesův teorém (Paradox vzácné nemoci)**
        Představte si, že test na nemoc má úžasnou přesnost (např. 99 %). Přesto, pokud je nemoc v populaci velmi vzácná, většina pozitivních výsledků testu je **falešných**.
        
        *   **Příklad ze života:** Testujeme 10 000 lidí. Nemoc má pouze 0.1 % populace (Zelené body). Test odhalí téměř všechny nemocné, ale protože testujeme 9 990 zdravých lidí s byť jen 1% chybovostí (falešná pozitivita), test mylně označí téměř 100 zdravých lidí za nemocné (Červené body). 
        *   **Výsledek:** I když vám vyjde pozitivní test, vaše skutečná šance, že nemoc máte, může být překvapivě velmi malá!
        """)
        
        prevalence = st.sidebar.slider('Výskyt nemoci v populaci (Prevalence v %):', 0.1, 10.0, 1.0, 0.1) / 100
        sensitivity = st.sidebar.slider('Senzitivita (Test najde nemocného v %):', 50.0, 99.9, 99.0, 0.1) / 100
        specificity = st.sidebar.slider('Specificita (Test pozná zdravého v %):', 50.0, 99.9, 99.0, 0.1) / 100
        
        N = 10000
        n_sick = int(N * prevalence)
        n_healthy = N - n_sick
        
        true_pos = int(n_sick * sensitivity)
        false_neg = n_sick - true_pos
        
        true_neg = int(n_healthy * specificity)
        false_pos = n_healthy - true_neg
        
        # Grid pro vizualizaci
        grid_size = 100
        x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        x = x.flatten()
        y = y.flatten()
        
        # Namapování barev (TP, FN, FP, TN)
        colors = (['forestgreen'] * true_pos + 
                  ['darkred'] * false_neg + 
                  ['crimson'] * false_pos + 
                  ['lightgray'] * true_neg)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(x, y, c=colors, s=25, alpha=0.9, edgecolors='none')
        
        # Výpočty pro titulek
        total_positive_tests = true_pos + false_pos
        prob_sick_if_positive = (true_pos / total_positive_tests) * 100 if total_positive_tests > 0 else 0
        
        title = f"Vzorek populace: {N} lidí\nPočet pozitivních testů: {total_positive_tests} lidí\nZ toho SKUTEČNĚ nemocní: {true_pos} lidí (Zeleně)\nFalešný poplach: {false_pos} lidí (Červeně)\n\nPravděpodobnost nemoci při pozitivním testu = {prob_sick_if_positive:.1f} %"
        
        plt.title(title, fontsize=14, fontweight='bold', color='darkslategray')
        plt.axis('off')
        
        # Legenda
        tp_patch = mpatches.Patch(color='forestgreen', label=f'Skutečně nemocní (Zelená): {true_pos}')
        fp_patch = mpatches.Patch(color='crimson', label=f'Falešný poplach (Červená): {false_pos}')
        tn_patch = mpatches.Patch(color='lightgray', label=f'Skutečně zdraví (Šedá): {true_neg}')
        fn_patch = mpatches.Patch(color='darkred', label=f'Nemocní, ale test je nezachytil (Tmavě červená): {false_neg}')
        plt.legend(handles=[tp_patch, fp_patch, fn_patch, tn_patch], loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=11)
        
        st.pyplot(fig)
        plt.close(fig)
