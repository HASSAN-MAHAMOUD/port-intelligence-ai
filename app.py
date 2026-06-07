import gradio as gr
import joblib
import pandas as pd
import numpy as np

pipe = joblib.load('meilleur_modele_foret_simple.pkl')

nom_cargaison = ['Fertilizer', 'Steel Bar', 'Steel Coil', 'Vehicle', 'Wheat',
                 'Cement', 'Sugar', 'Rice', 'Clinker', 'General Cargo']

nom_navire = sorted(pipe.named_steps['comp']
                .named_transformers_['cat']
                .named_steps['onehot']
                .categories_[1].tolist())

type_travail = sorted(pipe.named_steps['comp']
                .named_transformers_['cat']
                .named_steps['onehot']
                .categories_[4].tolist())

def predire(poids_tare, poids_cargaison, poids_sortie,
            heure, jour, mois, nom_carg, nom_nav, type_trav):

    op_entree = pipe.named_steps['comp'] \
                .named_transformers_['cat'] \
                .named_steps['imputer'] \
                .statistics_[2]
    op_sortie = pipe.named_steps['comp'] \
                .named_transformers_['cat'] \
                .named_steps['imputer'] \
                .statistics_[3]

    data = pd.DataFrame([{
        'Poids_Tare_kg'             : poids_tare,
        'Poids_Cargaison_kg'        : poids_cargaison,
        'Poids_Camion_Entree_kg'    : poids_tare + poids_cargaison,
        'Poids_Camion_Sortie_kg'    : poids_sortie,
        'heure_entree'              : heure,
        'jour_semaine'              : jour,
        'mois'                      : mois,
        'log_Poids_Tare_kg'         : np.log1p(poids_tare),
        'log_Poids_Cargaison_kg'    : np.log1p(poids_cargaison),
        'log_Poids_Camion_Sortie_kg': np.log1p(poids_sortie),
        'Nom_Cargaison'             : nom_carg,
        'Nom_Navire'                : nom_nav,
        'Operateur_Entree'          : op_entree,
        'Operateur_Sortie'          : op_sortie,
        'Type_Travail'              : type_trav,
    }])

    prediction = pipe.predict(data)[0]
    heures  = int(prediction // 60)
    minutes = int(prediction % 60)

    risque = "🔴 Risque élevé"  if prediction > 1440 else \
             "🟠 Risque modéré" if prediction > 600  else \
             "🟢 Risque faible"

    return (
        f"{prediction:.0f} minutes ({heures}h {minutes}min)",
        risque
    )

with gr.Blocks(title="PORT INTELLIGENCE AI — DMP") as app:

    # ── En-tête avec logo ──────────────────────────────────
    with gr.Row():
        gr.Image(
            value="logo_dmp.png",
            width=120,
            height=120,
            show_label=False,
            container=False
        )
        gr.Markdown(
            "# PORT INTELLIGENCE AI\n"
            "### Prédiction de la Surestarie — Doraleh Multi-Purpose Port"
        )

    # ── Formulaire ────────────────────────────────────────
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Poids (kg)")
            poids_tare      = gr.Number(label="Poids Tare",          value=15000)
            poids_cargaison = gr.Number(label="Poids Cargaison",      value=20000)
            poids_sortie    = gr.Number(label="Poids Camion Sortie",  value=35000)

        with gr.Column():
            gr.Markdown("#### Temporel")
            heure = gr.Slider(0, 23, value=14, step=1, label="Heure d'entrée")
            jour  = gr.Slider(0, 6,  value=1,  step=1, label="Jour (0=Lundi … 6=Dimanche)")
            mois  = gr.Slider(1, 12, value=3,  step=1, label="Mois")

        with gr.Column():
            gr.Markdown("#### Opération")
            nom_carg  = gr.Dropdown(nom_cargaison, label="Type de cargaison", value=nom_cargaison[0])
            nom_nav   = gr.Dropdown(nom_navire,    label="Navire")
            type_trav = gr.Dropdown(type_travail,  label="Type de travail")

    # ── Bouton et résultats ───────────────────────────────
    btn = gr.Button("Prédire la Surestarie", variant="primary")

    with gr.Row():
        out_duree  = gr.Textbox(label="Durée prédite")
        out_risque = gr.Textbox(label="Niveau de risque")

    btn.click(
        predire,
        inputs=[poids_tare, poids_cargaison, poids_sortie,
                heure, jour, mois, nom_carg, nom_nav, type_trav],
        outputs=[out_duree, out_risque]
    )

app.launch()

