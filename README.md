# Projet de Maintenance Prédictive - Dataset NASA CMAPSS

## Description
Ce projet illustre un cycle complet de data science appliqué à la maintenance prédictive dans le cadre de l'industrie 4.0.  
L'objectif n'est pas d'obtenir le modèle le plus performant, mais de démontrer la capacité à mettre en œuvre un pipeline complet allant de la préparation des données capteurs à la modélisation prédictive et la visualisation des résultats.

## Données
Les données utilisées proviennent du jeu de données public **NASA CMAPSS (Turbofan Engine Degradation Simulation Data Set)**.  
Elles contiennent des séries temporelles de capteurs mesurées sur plusieurs moteurs simulés jusqu'à leur panne.

- **Train** : cycles complets de fonctionnement des moteurs jusqu'à la panne.
- **Test** : séquences partielles de fonctionnement.
- **RUL (Remaining Useful Life)** : durée de vie résiduelle fournie pour les moteurs du test.

## Étapes du projet
1. **Préparation des données**
   - Nettoyage et mise en forme.
   - Construction de la RUL pour l’ensemble d’entraînement.
   - Normalisation et génération de séquences temporelles (pour LSTM).

2. **Exploration et visualisation**
   - Distribution des durées de vie des moteurs.
   - Analyse de la dégradation des capteurs.
   - Visualisations globales (scatter plots) et locales (suivi d’un moteur).

3. **Modélisation**
   - Baseline : RandomForest régressif.
   - RandomForest amélioré avec features dérivées (moyenne glissante, rolling features).
   - LSTM amélioré pour capturer les dynamiques temporelles.

4. **Évaluation**
   - Métriques utilisées : RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), R².
   - Comparaison des performances entre modèles.
   - Visualisation des erreurs par scatter plots et courbes de suivi moteur.

## Résultats principaux
- **RandomForest baseline** : MAE ≈ 35, RMSE ≈ 46.
- **RandomForest amélioré** : MAE ≈ 35, RMSE ≈ 47 (légère amélioration de la stabilité).
- **LSTM amélioré** : MAE ≈ 25, RMSE ≈ 35 (meilleur compromis).

Le LSTM prédit bien les pannes proches (RUL faible), mais a tendance à surestimer la durée de vie pour les moteurs encore jeunes.

## Conclusion
Ce projet démontre la mise en place d’un pipeline complet de maintenance prédictive :
- Préparation et transformation de données capteurs.
- Implémentation de modèles de machine learning et deep learning.
- Évaluation et visualisation des résultats.

Bien que les performances ne soient pas optimisées au maximum, ce travail illustre la maîtrise des étapes clés d’un projet de data science industriel.
