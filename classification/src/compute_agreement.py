from statsmodels.stats.inter_rater import fleiss_kappa
import os
import json
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, cohen_kappa_score

from config import *

# information
# https://github.com/Shamya/FleissKappa/blob/master/example_kappa.py
# https://www.statsmodels.org/stable/generated/statsmodels.stats.inter_rater.fleiss_kappa.html


##################################################################
# on final version of data
def load_expert():
    data_dir = os.path.join(covid_data_dir, "test", "expert")
    experts = ["biomedical_expert", "computer_science_expert"]

    final_result = {}
    for expert in experts:
        folder = os.path.join(data_dir, expert)
        filenames = [f for f in os.listdir(folder) if ".swp" not in f]

        res = {}
        for filename in filenames:
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as infile:
                data = json.load(infile)
                res[data["docId"]] = data["labels"]

        final_result[expert] = res
    return final_result

def load_crowd(id_list):
    data_dir = os.path.join(covid_data_dir, "test")
    
    res = {}
    for _id in id_list:
        with open(os.path.join(data_dir, "{}.json".format(_id)), 'r', encoding='utf-8') as infile:
            data = json.load(infile)

            res[data["paper_id"]] = [
                segment["crowd_label"]
                for paragraph in data["abstract"]
                for sent in paragraph["sentences"]
                for segment in sent
            ]

    return res

def compute():
    data = load_expert()
    article_id_list = sorted([k for k in data["biomedical_expert"].keys()])
    data["crowd"] = load_crowd(article_id_list)
        
    ##########################
    # single 
    result_table = np.zeros([len(data["biomedical_expert"]), 3])
    key_list = [
        ["biomedical_expert", "computer_science_expert"], 
        ["biomedical_expert", "crowd"], 
        ["computer_science_expert", "crowd"]
    ]
    for key_count, (key_1, key_2) in enumerate(key_list):
        data_1 = data[key_1]
        data_2 = data[key_2]

        all_matrix = []
        labels_1 = []
        labels_2 = []
        for article_count, article_id in enumerate(article_id_list):
            annotation_1 = data_1[article_id]
            annotation_2 = data_2[article_id]

            matrix = np.zeros([len(annotation_1), 5])
            for i, (a1, a2) in enumerate(zip(annotation_1, annotation_2)):
                # kappa
                matrix[i, label_mapping[a1]] += 1
                matrix[i, label_mapping[a2]] += 1

                # precision, recall, f1
                labels_1.append(label_mapping[a1])
                labels_2.append(label_mapping[a2])

            # kappa
            #kappa = fleiss_kappa(matrix)
            kappa = cohen_kappa_score(annotation_1, annotation_2)
            result_table[article_count, key_count] = kappa
            all_matrix.append(matrix)

            #if np.isnan(kappa):
            #    print(matrix)
            #    print(kappa)
            #    quit()
        

        # kappa
        all_matrix = np.vstack(all_matrix)
        all_kappa = fleiss_kappa(all_matrix)
        print("\n=={} vs {}==\nFleiss Kappa {} (size={})".format(key_1, key_2, all_kappa, str(all_matrix.shape)))

        # p, r, f
        score = precision_recall_fscore_support(labels_1, labels_2)
        output_score(score, "score-{}-{}.xlsx".format(key_1, key_2))
        acc = accuracy_score(labels_1, labels_2)
        print("Accuracy", acc)
        
        cohen = cohen_kappa_score(labels_1, labels_2)
        print("Cohen", cohen)


    #print(result_table)

    # output
    result_table = pd.DataFrame(result_table, index=article_id_list, columns=["{} - {}".format(k1, k2) for k1, k2 in key_list])
    result_table = result_table.fillna(value="Nan")
    result_table.to_excel(os.path.join(result_dir, "kappa.xlsx"))

def output_score(score, filename):
    table = pd.DataFrame(
        [score[3], score[0], score[1], score[2]],
        index=["# samples", "Precision", "Recall", "F1"],
        columns=["background", "purpose", "method", "finding", "other"],
    )
    table.to_excel(os.path.join(result_dir, filename))

def main():
    compute()

if __name__ == "__main__":
    main()



