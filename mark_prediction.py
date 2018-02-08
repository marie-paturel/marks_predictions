# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:44:09 2018

@author: Marie Paturel Quitoque
"""

###############################################
########## Installation des packages ##########
###############################################
import os
import time
import pandas as pd
import numpy as np
# from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#########################################################
########## Lecture et preparation des donnees ###########
#########################################################

DOS = r"C:\Users\Quitoque\Documents\Data\Tests_2018_01_30"

##### article #####
ARTICLE_PATH = os.path.join(DOS, "article.csv")
article = pd.read_csv(ARTICLE_PATH, sep=";", decimal=".", encoding='latin-1',
                      header=0).rename(columns={'id': 'article_id'})


##### mark #####
MARK_PATH = os.path.join(DOS, "mark.csv")
mark = pd.read_csv(MARK_PATH, sep=";", decimal=".", encoding='latin-1',
                   header=0).rename(columns={'id': 'mark_id'})
mark_recipe = mark.drop(mark[pd.isnull(mark.recipe_id)].index)

##### recipe #####
RECIPE_PATH = os.path.join(DOS, "recipe.csv")
recipe = pd.read_csv(RECIPE_PATH, sep=";", decimal=".", encoding='latin-1',
                     header=0).rename(columns={'id': 'recipe_id'})

##### recipe_info #####
RECIPE_INFO_PATH = os.path.join(DOS, "recipe_info.csv")
recipe_info = pd.read_csv(RECIPE_INFO_PATH, sep=";", decimal=".", encoding='latin-1',
                          header=0).rename(columns={'id': 'recipe_info_id'})

# Creation d'une table cooking_time et d'une table waiting_time
cooking_time = recipe_info.query('type == "cooking_time"').rename(
    columns={'recipe_info_id': 'cooking_time_id'})
waiting_time = recipe_info.query('type == "waiting_time"').rename(
    columns={'recipe_info_id': 'waiting_time_id'})
waiting_time.index = cooking_time.index


# Ajout d'une colonne avec le temps moyen
cooking_time["cooking_time"] = [np.mean([int([cooking_time.name[a].split(
    )[i] for i in [0, 2]][j]) for j in [0, 1]]) for a in range(len(cooking_time.index))]
waiting_time["waiting_time"] = [np.mean([int([waiting_time.name[a].split(
    )[i] for i in [0, 2]][j]) for j in [0, 1]]) for a in range(len(waiting_time.index))]

##### recipe_ingredient #####
RECIPE_INGREDIENT_PATH = os.path.join(DOS, "recipe_ingredient.csv")
recipe_ingredient = pd.read_csv(RECIPE_INGREDIENT_PATH, sep=";", decimal=".", encoding='latin-1',
                                header=0).rename(columns={'id': 'recipe_ingredient_id'})


##### recipe_recipe_info_cookingtools #####
RECIPE_RECIPE_INFO_COOKINGTOOLS_PATH = os.path.join(DOS, "recipe_recipe_info_cookingtools.csv")
recipe_recipe_info_cookingtools = pd.merge(
    pd.read_csv(RECIPE_RECIPE_INFO_COOKINGTOOLS_PATH, sep=";", decimal=".",
                encoding='latin-1', header=0),
    recipe_info, how="left")


##### recipe_recipe_info_cupboardtools #####
RECIPE_RECIPE_INFO_CUPBOARDTOOLS_PATH = os.path.join(DOS, "recipe_recipe_info_cupboardtools.csv")
recipe_recipe_info_cupboardtools = pd.merge(
    pd.read_csv(RECIPE_RECIPE_INFO_CUPBOARDTOOLS_PATH, sep=";", decimal=".",
                encoding='latin-1', header=0),
    recipe_info, how="left")


##### subscription_order #####
SUBSCRIPTION_ORDER_PATH = os.path.join(DOS, "subscription_order.csv")
subscription_order = pd.read_csv(SUBSCRIPTION_ORDER_PATH, sep=";", decimal=".",
                                 encoding='latin-1', header=0,
                                 dtype={'postal_code' : object, 'phone': object,
                                        'created_script' : object}).rename(
                                            columns={'id': 'subscription_order_id'})

##### user #####
USER_PATH = os.path.join(DOS, "user.csv")
user = pd.read_csv(USER_PATH, sep=";", decimal=".", encoding='latin-1',
                   header=0, dtype={'created_script' : object}).rename(
                       columns={'id': 'user_id'})

#### INSEE #####
DOS_INSEE = r'C:\Users\Quitoque\Documents\Data\INSEE\table-appartenance-geo-communes-17'
INSEE_PATH = os.path.join(DOS_INSEE, "INSEE.csv")
INSEE = pd.read_csv(INSEE_PATH, sep=";", decimal=".", encoding='latin-1',
                    header=0, usecols=['postal_code', 'Région', "Tranche d'unité urbaine 2014"],
                    dtype={'postal_code' : object}).rename(
                       columns={'Région' : 'region',
                                "Tranche d'unité urbaine 2014" : "tranche2014"}).drop_duplicates()
INSEE.postal_code = INSEE.postal_code.str.rjust(5, "0")

#CP_PATH = os.path.join(DOS_INSEE, "laposte_hexasmal.csv")
#CP = pd.read_csv(CP_PATH, sep=";", decimal=".", encoding='latin-1',
#                   header=0, dtype={'Code_postal' : object,
#                                    'Code_commune_INSEE' : object}).rename(
#                       columns={'Code_commune_INSEE': 'code_commune',
#                                'Code_postal' : 'postal_code'})
#CP.code_commune = CP.code_commune.str.rjust(5, "0")
#CP.postal_code = CP.postal_code.str.rjust(5, "0")




#INSEE.groupby('postal_code').first().to_csv("INSEE_first.csv", decimal=',', sep=';')
#INSEE.groupby('postal_code').mean().to_csv("INSEE_mean.csv", decimal=',', sep=';')
#INSEE.groupby('postal_code').median().to_csv("INSEE_median.csv", decimal=',', sep=';')

INSEE = INSEE.groupby('postal_code').median()
INSEE['postal_code'] = INSEE.index

#########################################
########## Prediction de notes ##########
#########################################

# Suppression des lignes sans user_id
mark_recipe = mark_recipe.drop(mark_recipe[(pd.isnull(
    mark_recipe.user_id) | pd.isnull(mark_recipe.subscription_order_id))].index)

# Jointure des marks avec user, subscritpion_order, recipe, cooking_time, waiting_time
mark_recipe_all = mark_recipe[["mark_id", "user_id", "subscription_order_id", "recipe_id",
                               "mark"]].merge(
                                   user[["user_id", "birthday"]],
                                   on='user_id', how="left").merge(
                                       subscription_order[["subscription_order_id",
                                                           "delivery_day_id",
                                                           "delivery_time_slot_id", "delivery_date",
                                                           "postal_code", "total_box_price_ttc",
                                                           "total_discount", "delivery_charge_ttc",
                                                           "total_ttc", "paid_price_ttc",
                                                           "first"]],
                                       on='subscription_order_id', how="left").merge(
                                           recipe[["recipe_id", "box_week_id",
                                                   "cooking_time_id", "waiting_time_id",
                                                   "recipe_text"]],
                                           on="recipe_id", how="left").merge(
                                               cooking_time[["cooking_time_id", "cooking_time"]],
                                               on="cooking_time_id", how="left").merge(
                                                   waiting_time[["waiting_time_id",
                                                                 "waiting_time"]],
                                                   on="waiting_time_id", how="left").merge(
                                                       INSEE, how='left',
                                                       on='postal_code')


# Calcul de l'age des clients
mark_recipe_all['delivery_date'] = pd.to_datetime(mark_recipe_all['delivery_date'],
                                                  format='%Y-%m-%d')
mark_recipe_all['birthday'] = pd.to_datetime(mark_recipe_all['birthday'], format='%Y-%m-%d')
mark_recipe_all['age'] = (mark_recipe_all['delivery_date'] -
                          mark_recipe_all['birthday'])
mark_recipe_all['age'] = [mark_recipe_all['age'][a].days / 365.25
                          for a in mark_recipe_all.index]

# Gestion des ages mal renseignes
# Calcul de la moyenne des ages
MEAN_AGE = np.mean(mark_recipe_all[(mark_recipe_all.age > 15) & (mark_recipe_all.age < 100)].age)
SD_AGE = np.std(mark_recipe_all[(mark_recipe_all.age > 15) & (mark_recipe_all.age < 100)].age)

# Remplacement des ages extremes par des valeurs issues d'une loi normale de moyenne MEAN_AGE
mark_recipe_all.loc[((mark_recipe_all.age < 15) | (mark_recipe_all.age > 100)),
                    'age'] = np.random.randn(sum(
                        (mark_recipe_all.age < 15) | (mark_recipe_all.age > 100))) * \
                        SD_AGE + MEAN_AGE

# Ajout d'une colonne mois de livraison
mark_recipe_all['month'] = [mark_recipe_all.delivery_date[a].month
                            for a in mark_recipe_all.index]

# Ajout d'une colonne departement
mark_recipe_all['departement'] = [str(mark_recipe_all.postal_code[a])[:2]
                                  for a in mark_recipe_all.index]

# Ajout d'une colonne Province
IDF = ["75", "77", "78", "91", "92", "93", "94", "95"]
mark_recipe_all['province'] = [mark_recipe_all['departement'][a] in IDF
                               for a in mark_recipe_all.index]

# Ajout d'une colonne avec nombre de caracteres de la recette ?
mark_recipe_all['nb_char_recette'] = [len(mark_recipe_all['recipe_text'][a])
                                      for a in mark_recipe_all.index]

# Creation d'un dataframe avec une colonne par ingredient
mat_recipe_ingredient = pd.crosstab(recipe_ingredient['recipe_id'],
                                    recipe_ingredient['ingredient_id'])

mat_recipe_ingredient.columns = ['ing_' + str(col) for col in mat_recipe_ingredient.columns]
mat_recipe_ingredient['recipe_id'] = mat_recipe_ingredient.index

N_AV_ING = len(mark_recipe_all.columns)

mark_recipe_all = pd.merge(mark_recipe_all, mat_recipe_ingredient, on="recipe_id", how="left")

N_AP_ING = len(mark_recipe_all.columns)

mark_recipe_all['n_ing'] = mark_recipe_all.iloc[:, N_AV_ING:N_AP_ING].sum(axis=1)


# Creation d'un dataframe avec une colonne par cookingtools
mat_recipe_cookingtools = pd.crosstab(recipe_recipe_info_cookingtools['recipe_id'],
                                      recipe_recipe_info_cookingtools['recipe_info_id'])

mat_recipe_cookingtools.columns = ['ckt_' + str(col) for col in mat_recipe_cookingtools.columns]
mat_recipe_cookingtools['recipe_id'] = mat_recipe_cookingtools.index

N_AV_CKT = len(mark_recipe_all.columns)

mark_recipe_all = pd.merge(mark_recipe_all, mat_recipe_cookingtools, on="recipe_id",
                           how="left")

N_AP_CKT = len(mark_recipe_all.columns)

mark_recipe_all['n_ckt'] = mark_recipe_all.iloc[:, N_AV_CKT:N_AP_CKT].sum(axis=1)


# Creation d'un dataframe avec une colonne par cupboardtools
mat_recipe_cupboardtools = pd.crosstab(recipe_recipe_info_cupboardtools['recipe_id'],
                                       recipe_recipe_info_cupboardtools['recipe_info_id'])

mat_recipe_cupboardtools.columns = ['cbt_' + str(col) for col in mat_recipe_cupboardtools.columns]
mat_recipe_cupboardtools['recipe_id'] = mat_recipe_cupboardtools.index

N_AV_CBT = len(mark_recipe_all.columns)

mark_recipe_all = pd.merge(mark_recipe_all, mat_recipe_cupboardtools, on="recipe_id",
                           how="left")

N_AP_CBT = len(mark_recipe_all.columns)

mark_recipe_all['n_cbt'] = mark_recipe_all.iloc[:, N_AV_CBT:N_AP_CBT].sum(axis=1)

mark_recipe_all.n_cbt.value_counts()


# Suppression des colonnes inutiles
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


# Remplacement des NA par des 0
mark_recipe_all = mark_recipe_all.fillna(0)

# Transformation en facteurs
mark_recipe_all.delivery_day_id = mark_recipe_all.delivery_day_id.astype("category")
mark_recipe_all.delivery_time_slot_id = mark_recipe_all.delivery_time_slot_id.astype("category")
mark_recipe_all.month = mark_recipe_all.month.astype("category")

######################################
########## Modele Predictif ##########
######################################

##### Jeux de donnees train et test #####
m = mark_recipe_all.drop(['mark'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(mark_recipe_all, mark_recipe_all.mark,
                                                    test_size=0.4, random_state=0)
#tps1 = time.time()
#
#
#tps2 = time.time()

##### Temps
# 100 + 100 : 0.03
# 1000 + 100 : 1,11
# 1000 + 500 : 1,52
# 5000 + 100 : 25,1
# 5000 + 500 : 28,54
# 10000 + 100 : 103,3
# 10000 + 500 : 103,8
# 10000 + 1000 : 105,5
# 20000 + 1000 : 471,60
# 50000 + 10000 : 471,60


# dict with optimal models
models = {}
# find optimal value of alpha 
n_trials = 100
alpha_list = 10 ** np.linspace(-5, 5, n_trials)
# number of folds for cross validation
cv = 5
# find optimal value of l1 (for ElasticNet)
l1_list = 10 ** np.linspace(-2, 0, 50)
max_iter = 5000


# #############################################################################
# Lasso
ALPHA = 0.1
lasso = Lasso(alpha=ALPHA)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
R2_SCORE_LASSO = r2_score(y_test, y_pred_lasso)
print(lasso)
print("r^2 on test data : %f" % R2_SCORE_LASSO)

# #############################################################################
# ElasticNet
enet = ElasticNet(alpha=ALPHA, l1_ratio=0.7)

y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
R2_SCORE_ENET = r2_score(y_test, y_pred_enet)
print(enet)
print("r^2 on test data : %f" % R2_SCORE_ENET)

# #############################################################################
# ElasticNet
ridge = Ridge(alpha=ALPHA, l1_ratio=0.7)

y_pred_ridge = ridge.fit(X_train, y_train).predict(X_test)
R2_SCORE_RIDGE = r2_score(y_test, y_pred_ridge)
print(ridge)
print("r^2 on test data : %f" % R2_SCORE_RIDGE)