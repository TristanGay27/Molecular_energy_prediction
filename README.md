# Molecular Energy Prediction 

Projet de 5ModIA réalisé par Tristan Gay et Clément Gris 

## Description

Ce projet se concentre sur la prédiction de la surface d'énergie potentielle interatomique pour des molécules organiques de petite taille. Notre but est de développer un modèle capable de prédire l'énergie d'atomisation pour une molécule donnée, en utilisant des informations géométriques et des données supplémentaires sur les atomes. 

## Informations générales

- Prédiction de l'énergie des molécules.
- Respect des contraintes de symétrie (translation, rotation, permutation). 
- Utilisation de l'ensemble de données QM7-X pour l'entraînement et les tests. 
- Évaluation des modèles à l'aide de l'erreur quadratique moyenne (RMSE). 

## Structure du Projet

| Dossier/Fichier | Description |
|-----------------|-------------|
| `data_exploration` | Exploration des données |
| `models` | Modèles  |
| `models_scattering` | Modèles de scattering |
| `results` | Stockage des résultats |
| `TP_Bayes_optimal_NN.ipynb` | Notebook pour le produit ElementWise |
| `utils_project.py` | Script de gestion des données |
