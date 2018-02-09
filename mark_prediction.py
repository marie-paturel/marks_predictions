# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:44:09 2018

@author: Marie Paturel Quitoque
"""

###############################################################################
######################## Chargement des packages ##############################
###############################################################################
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split, KFold


###############################################################################
################## Lecture et preparation des donnees #########################
###############################################################################

DOS = r"C:\Users\Quitoque\Documents\Data\Tests_2018_01_30"

################################ article ######################################
# Lecture du fichier article.csv
ARTICLE_PATH = os.path.join(DOS, "article.csv")
article = pd.read_csv(ARTICLE_PATH, sep=";", decimal=".", encoding='latin-1',
                      header=0).rename(columns={'id': 'article_id'})

################################ mark #########################################
# Lecture du fichier mark.csv
MARK_PATH = os.path.join(DOS, "mark.csv")
mark = pd.read_csv(MARK_PATH, sep=";", decimal=".", encoding='latin-1',
                   usecols=["id", "user_id", "subscription_order_id",
                            "recipe_id", "mark"],
                   header=0).rename(columns={'id': 'mark_id'})
mark_recipe = mark.drop(mark[pd.isnull(mark.recipe_id)].index)

################################ recipe #######################################
# Lecture du fichier recipe.csv
RECIPE_PATH = os.path.join(DOS, "recipe.csv")
recipe = pd.read_csv(RECIPE_PATH, sep=";", decimal=".", encoding='latin-1',
                     usecols=["id", "box_week_id", "cooking_time_id",
                              "waiting_time_id", "recipe_text"],
                     header=0).rename(columns={'id': 'recipe_id'})

################################ recipe_info ##################################
# Lecture du fichier recipe_info.csv
RECIPE_INFO_PATH = os.path.join(DOS, "recipe_info.csv")
recipe_info = pd.read_csv(RECIPE_INFO_PATH, sep=";", decimal=".", encoding='latin-1',
                          header=0).rename(columns={'id': 'recipe_info_id'})

# Creation d'une table cooking_time et d'une table waiting_time depuis recipe_info
cooking_time = recipe_info.query('type == "cooking_time"').rename(
    columns={'recipe_info_id': 'cooking_time_id'})
waiting_time = recipe_info.query('type == "waiting_time"').rename(
    columns={'recipe_info_id': 'waiting_time_id'})
waiting_time.index = cooking_time.index

# Ajout d'une colonne avec le temps moyen pour chaque intervalle
cooking_time["cooking_time"] = [np.mean([int([cooking_time.name[a].split(
    )[i] for i in [0, 2]][j]) for j in [0, 1]]) for a in range(len(cooking_time.index))]
waiting_time["waiting_time"] = [np.mean([int([waiting_time.name[a].split(
    )[i] for i in [0, 2]][j]) for j in [0, 1]]) for a in range(len(waiting_time.index))]

del cooking_time["name"]
del cooking_time["type"]
del waiting_time["name"]
del waiting_time["type"]

################################ recipe_ingredient ############################
# Lecture du fichier recipe_ingredient.csv
RECIPE_INGREDIENT_PATH = os.path.join(DOS, "recipe_ingredient.csv")
recipe_ingredient = pd.read_csv(RECIPE_INGREDIENT_PATH, sep=";", decimal=".", encoding='latin-1',
                                header=0).rename(columns={'id': 'recipe_ingredient_id'})


################################ recipe_recipe_info_cookingtools ##############
# Lecture du fichier recipe_recipe_info_cookingtools.csv
RECIPE_RECIPE_INFO_COOKINGTOOLS_PATH = os.path.join(DOS, "recipe_recipe_info_cookingtools.csv")
# Jointure entre cette table et recipe_info par recipe_info_id
recipe_recipe_info_cookingtools = pd.merge(
    pd.read_csv(RECIPE_RECIPE_INFO_COOKINGTOOLS_PATH, sep=";", decimal=".",
                encoding='latin-1', header=0),
    recipe_info, how="left")


################################ recipe_recipe_info_cupboardtools #############
# Lecture du fichier recipe_recipe_info_cupboardtools.csv
RECIPE_RECIPE_INFO_CUPBOARDTOOLS_PATH = os.path.join(DOS, "recipe_recipe_info_cupboardtools.csv")
# Jointure entre cette table et recipe_info par recipe_info_id
recipe_recipe_info_cupboardtools = pd.merge(
    pd.read_csv(RECIPE_RECIPE_INFO_CUPBOARDTOOLS_PATH, sep=";", decimal=".",
                encoding='latin-1', header=0),
    recipe_info, how="left")


################################ subscription_order ###########################
# Lecture du fichier subscription_order.csv
SUBSCRIPTION_ORDER_PATH = os.path.join(DOS, "subscription_order.csv")
subscription_order = pd.read_csv(SUBSCRIPTION_ORDER_PATH, sep=";", decimal=".",
                                 usecols=["id", "delivery_day_id", "delivery_time_slot_id",
                                          "delivery_date", "postal_code", "total_box_price_ttc",
                                          "total_discount", "delivery_charge_ttc",
                                          "total_ttc", "paid_price_ttc", "first"],
                                 encoding='latin-1', header=0,
                                 dtype={'id' : object, 'delivery_day_id' : object,
                                        'delivery_time_slot_id' : object, 'delivery_date' : object,
                                        'postal_code' : object, 'total_box_price_ttc' : float,
                                        'total_discount' : float, 'delivery_charge_ttc' : float,
                                        'total_ttc' : float, 'paid_price_ttc' : float,
                                        'first' : object}).rename(
                                            columns={'id': 'subscription_order_id'})

################################ user #########################################
# Lecture du fichier user.csv
USER_PATH = os.path.join(DOS, "user.csv")
user = pd.read_csv(USER_PATH, sep=";", decimal=".", encoding='latin-1',
                   usecols=["id", "birthday"],
                   header=0).rename(columns={'id': 'user_id'})

################################ INSEE ########################################
# Lecture du fichier INSEE.csv
DOS_INSEE = r'C:\Users\Quitoque\Documents\Data\INSEE\table-appartenance-geo-communes-17'
INSEE_PATH = os.path.join(DOS_INSEE, "INSEE.csv")
INSEE = pd.read_csv(INSEE_PATH, sep=";", decimal=".", encoding='latin-1',
                    header=0, usecols=['postal_code', 'Région', "Tranche d'unité urbaine 2014"],
                    dtype={'postal_code' : object}).rename(
                        columns={'Région' : 'region',
                                 "Tranche d'unité urbaine 2014" : "tranche2014"}).drop_duplicates()
# Ajout d'un 0 pour les codes postaux a 4 chiffres
INSEE.postal_code = INSEE.postal_code.str.rjust(5, "0")

#INSEE.groupby('postal_code').first().to_csv("INSEE_first.csv", decimal=',', sep=';')
#INSEE.groupby('postal_code').mean().to_csv("INSEE_mean.csv", decimal=',', sep=';')
#INSEE.groupby('postal_code').median().to_csv("INSEE_median.csv", decimal=',', sep=';')

# Pour les codes postaux avec plusieurs valeurs, on prend la mediane
INSEE = INSEE.groupby('postal_code').median()
# Creation d'une colonne avec les codes postaux a partir des index
INSEE['postal_code'] = INSEE.index


###############################################################################
##################  Creation du dataframe pour l'etude ########################
###############################################################################

# Suppression des lignes de mark_recipe sans user_id ni subscription_order_id
mark_recipe = mark_recipe.drop(mark_recipe[(pd.isnull(
    mark_recipe.user_id) | pd.isnull(mark_recipe.subscription_order_id))].index)

# Jointure des marks avec user, subscritpion_order, recipe, cooking_time, waiting_time, INSEE
mark_recipe_all = mark_recipe.merge(user, on='user_id', how="left").merge(
    subscription_order, on='subscription_order_id', how="left").merge(
        recipe, on="recipe_id", how="left").merge(
            cooking_time, on="cooking_time_id", how="left").merge(
                waiting_time, on="waiting_time_id", how="left").merge(
                    INSEE, on='postal_code', how='left')


################################ Age des clients ##############################
# Passage au format date
mark_recipe_all['delivery_date'] = pd.to_datetime(mark_recipe_all['delivery_date'],
                                                  format='%Y-%m-%d')
mark_recipe_all['birthday'] = pd.to_datetime(mark_recipe_all['birthday'], format='%Y-%m-%d')
# Calcul de la difference en nombre de jours
mark_recipe_all['age'] = (mark_recipe_all['delivery_date'] -
                          mark_recipe_all['birthday'])
# Calcul de l'age
mark_recipe_all['age'] = [mark_recipe_all['age'][a].days / 365.25
                          for a in mark_recipe_all.index]

################################ Gestion des ages mal renseignes ##############
# Calcul de la moyenne et de l'ecart-type des ages entre 15 et 100 ans
MEAN_AGE = np.mean(mark_recipe_all[(mark_recipe_all.age > 15) & (mark_recipe_all.age < 100)].age)
SD_AGE = np.std(mark_recipe_all[(mark_recipe_all.age > 15) & (mark_recipe_all.age < 100)].age)

# Remplacement des ages extremes par des valeurs issues de la loi normale MEAN_AGE, SD_AGE
mark_recipe_all.loc[((mark_recipe_all.age < 15) | (mark_recipe_all.age > 100)),
                    'age'] = np.random.randn(sum(
                        (mark_recipe_all.age < 15) | (mark_recipe_all.age > 100))) * \
                        SD_AGE + MEAN_AGE

################################ Mois de livraison ############################
# Ajout d'une colonne mois de livraison a partir de la date de livraison
mark_recipe_all['month'] = [mark_recipe_all.delivery_date[a].month
                            for a in mark_recipe_all.index]

################################ Province ou pas ##############################
# Ajout d'une colonne departement a partir du code postal
mark_recipe_all['departement'] = [str(mark_recipe_all.postal_code[a])[:2]
                                  for a in mark_recipe_all.index]

# Creation d'une liste avec les departements d'Ile de France
IDF = ["75", "77", "78", "91", "92", "93", "94", "95"]

# Creation de la colonne province 0/1
mark_recipe_all['province'] = [mark_recipe_all['departement'][a] in IDF
                               for a in mark_recipe_all.index]

################################ Difficulte de la recette #####################
# Ajout d'une colonne avec nombre de caracteres de la recette
mark_recipe_all['nb_char_recette'] = [len(mark_recipe_all['recipe_text'][a])
                                      for a in mark_recipe_all.index]

################################ Ingredients ##################################
# Creation d'un dataframe avec une colonne par ingredient et une ligne par recette
mat_recipe_ingredient = pd.crosstab(recipe_ingredient['recipe_id'],
                                    recipe_ingredient['ingredient_id'])
# Ajout d'un prefixe
mat_recipe_ingredient.columns = ['ing_' + str(col) for col in mat_recipe_ingredient.columns]

# Creation d'une colonne avec l'id de la recette pour faire la jointure
mat_recipe_ingredient['recipe_id'] = mat_recipe_ingredient.index

# Calcul du nombre de colonnes avant jointure
N_AV_ING = len(mark_recipe_all.columns)

# Jointure entre les notes et les ingredients par recette
mark_recipe_all = pd.merge(mark_recipe_all, mat_recipe_ingredient, on="recipe_id", how="left")

# Calcul du nombre de colonnes apres jointure
N_AP_ING = len(mark_recipe_all.columns)

# Calcul du nombre d'ingredients pour chaque recette
mark_recipe_all['n_ing'] = mark_recipe_all.iloc[:, N_AV_ING:N_AP_ING].sum(axis=1)

################################ cookingtools #################################
# Creation d'un dataframe avec une colonne par cookingtools et une ligne par recette
mat_recipe_cookingtools = pd.crosstab(recipe_recipe_info_cookingtools['recipe_id'],
                                      recipe_recipe_info_cookingtools['recipe_info_id'])
# Ajout d'un prefixe
mat_recipe_cookingtools.columns = ['ckt_' + str(col) for col in mat_recipe_cookingtools.columns]

# Creation d'une colonne avec l'id de la recette pour faire la jointure
mat_recipe_cookingtools['recipe_id'] = mat_recipe_cookingtools.index

# Calcul du nombre de colonnes avant jointure
N_AV_CKT = len(mark_recipe_all.columns)

# Jointure entre les notes et les ingredients par recette
mark_recipe_all = pd.merge(mark_recipe_all, mat_recipe_cookingtools, on="recipe_id",
                           how="left")

# Calcul du nombre de colonnes apres jointure
N_AP_CKT = len(mark_recipe_all.columns)

# Calcul du nombre de cookingtools pour chaque recette
mark_recipe_all['n_ckt'] = mark_recipe_all.iloc[:, N_AV_CKT:N_AP_CKT].sum(axis=1)


################################ cupboardtools ################################
# Creation d'un dataframe avec une colonne par cupboardtools et une ligne par recette
mat_recipe_cupboardtools = pd.crosstab(recipe_recipe_info_cupboardtools['recipe_id'],
                                       recipe_recipe_info_cupboardtools['recipe_info_id'])
# Ajout d'un prefixe
mat_recipe_cupboardtools.columns = ['cbt_' + str(col) for col in mat_recipe_cupboardtools.columns]

# Creation d'une colonne avec l'id de la recette pour faire la jointure
mat_recipe_cupboardtools['recipe_id'] = mat_recipe_cupboardtools.index

# Calcul du nombre de colonnes avant jointure
N_AV_CBT = len(mark_recipe_all.columns)

# Jointure entre les notes et les ingredients par recette
mark_recipe_all = pd.merge(mark_recipe_all, mat_recipe_cupboardtools, on="recipe_id",
                           how="left")

# Calcul du nombre de colonnes apres jointure
N_AP_CBT = len(mark_recipe_all.columns)

# Calcul du nombre de cupboardtools pour chaque recette
mark_recipe_all['n_cbt'] = mark_recipe_all.iloc[:, N_AV_CBT:N_AP_CBT].sum(axis=1)

################################ Suppression de colonnes ######################
del mark_recipe_all['cooking_time_id']
del mark_recipe_all['waiting_time_id']
del mark_recipe_all['birthday']
del mark_recipe_all['delivery_date']
del mark_recipe_all['postal_code']
del mark_recipe_all['departement']
del mark_recipe_all['recipe_text']
del mark_recipe_all['mark_id']
del mark_recipe_all['user_id']
del mark_recipe_all['subscription_order_id']
del mark_recipe_all['recipe_id']
del mark_recipe_all['box_week_id']

################################ Remplacement des NA par des 0 ################
mark_recipe_all = mark_recipe_all.fillna(0)

################################ Transformation en categories #################
mark_recipe_all.delivery_day_id = mark_recipe_all.delivery_day_id.astype("category")
mark_recipe_all.delivery_time_slot_id = mark_recipe_all.delivery_time_slot_id.astype("category")
mark_recipe_all.month = mark_recipe_all.month.astype("category")

###############################################################################
###########################  Modeles predicts  ################################
###############################################################################

y = mark_recipe_all.mark
X = mark_recipe_all.drop(labels='mark', axis=1)

################################ Train et test set ############################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#tps1 = time.time()
#
#
#tps2 = time.time()

################################ Recherche des modeles optimaux ###############

# #############################################################################
# Ridge
ridge = Ridge(alpha=0.1)
model_ridge = ridge.fit(X_train, y_train)

y_pred_ridge = model_ridge.predict(X_test)

coef_ridge = pd.Series(model_ridge.coef_, index=X_train.columns).sort_values()
imp_coef = pd.concat([coef_ridge.head(10), coef_ridge.tail(10)])
imp_coef.plot(kind="barh")
plt.title("Coefficients in the Model Ridge")

R2_SCORE_RIDGE = r2_score(y_test, y_pred_ridge)
print(ridge)
print("r^2 on test data : %f" % R2_SCORE_RIDGE)

# #############################################################################
# Lasso
lasso = Lasso(alpha=0.1)
model_lasso = lasso.fit(X_train, y_train)

y_pred_lasso = model_lasso.predict(X_test)

coef_lasso = pd.Series(model_lasso.coef_, index=X_train.columns).sort_values()
imp_coef_lasso = pd.concat([coef_lasso.head(10), coef_lasso.tail(10)])
imp_coef_lasso.plot(kind="barh")
plt.title("Coefficients in the Model Lasso")

R2_SCORE_LASSO = r2_score(y_test, y_pred_lasso)
print(lasso)
print("r^2 on test data : %f" % R2_SCORE_LASSO)

# #############################################################################
# ElasticNet
enet = ElasticNet(alpha=0.1, l1_ratio=0.7)
model_enet = enet.fit(X_train, y_train)

y_pred_enet = model_enet.predict(X_test)

coef_enet = pd.Series(model_enet.coef_, index=X_train.columns).sort_values()
imp_coef_enet = pd.concat([coef_enet.head(10), coef_lasso.tail(10)])
imp_coef_enet.plot(kind="barh")
plt.title("Coefficients in the Model Enet")

R2_SCORE_ENET = r2_score(y_test, y_pred_enet)
print(enet)
print("r^2 on test data : %f" % R2_SCORE_ENET)
