import os
import pandas as pd
import numpy as np
from sklearn import metrics
import functions.evaluation.prediction as pre


def evaluation(model, training_name, mask, df_name, training_gender):
    auc = []

    # VALIDACIÓN sobre cada género
    for df_val in ['NIH_male_val.csv', 'NIH_female_val.csv']:
        df_path = os.path.join("./results/gender_bias/prediction_dfs", df_val)

        # Cargamos el df de validación
        df_val = pd.read_csv(df_path)
        img_paths_val = [os.path.join('/home/mr1142/Documents/Data/folder_NIH',mask, img) for img in df_val.img_names]

        # Cogemos y real e y predicho
        y_pred = pre.prediction_tensor(model, img_paths_val, batch_size = 50)
        y_true = np.array(df_val.normal)

        # Guardamos la prediccion
        df_val[mask +"_"+ training_gender + "_" + training_name] = y_pred
        df_val.to_csv(df_path, index = False)

        # Calculamos AUC
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
        auc.append(metrics.auc(fpr, tpr))
    
    results = pd.read_csv("./results/gender_bias/gender_results.csv")
    results.loc[len(results)] = [training_name, df_name, mask] + auc
    results.reset_index(drop = True)
    results.to_csv("./results/gender_bias/gender_results.csv", index = False)