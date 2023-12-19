import pickle
import csv
import numpy as np
import os
import re
import xml.etree.ElementTree as ET
import time
import networkx as nx
import community
import  xml.dom.minidom
import xml.etree.ElementTree as ET
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import mean_squared_log_error,accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import connected_components
import sys


# Defina o caminho para salvar os dados dos clusters e das representações
save_path = "results/clusters_and_embeddings/"

if not os.path.exists(save_path):
    os.makedirs(save_path)

with open("gene/PHNet.pkl", 'rb') as file:
    PHNet = pickle.load(file)

with open('final_emb/pemb_final.pkl', "rb") as file_obj:
    pembd = pickle.load(file_obj)

def pairwise_evaluate(correct_labels, pred_labels):
    TP = 0.0  # Pairs Correctly Predicted To SameAuthor
    TP_FP = 0.0  # Total Pairs Predicted To SameAuthor
    TP_FN = 0.0  # Total Pairs To SameAuthor

    for i in range(len(correct_labels)):
        for j in range(i + 1, len(correct_labels)):
            if correct_labels[i] == correct_labels[j]:
                TP_FN += 1
            if pred_labels[i] == pred_labels[j]:
                TP_FP += 1
            if (correct_labels[i] == correct_labels[j]) and (pred_labels[i] == pred_labels[j]):
                TP += 1

    if TP == 0:
        pairwise_precision = 0
        pairwise_recall = 0
        pairwise_f1 = 0
    else:
        pairwise_precision = TP / TP_FP
        pairwise_recall = TP / TP_FN
        pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (pairwise_precision + pairwise_recall)
    return pairwise_precision, pairwise_recall, pairwise_f1

def calculate_ACP_AAP(correct_labels, cluster_labels):
    unique_clusters = np.unique(cluster_labels)
    ACP = 0.0
    AAP = 0.0

    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_author_labels = correct_labels[cluster_indices]
        unique_author_labels, author_counts = np.unique(cluster_author_labels, return_counts=True)

        # Calculate ACP
        max_count = np.max(author_counts)
        ACP += max_count / len(cluster_indices)

        # Calculate AAP
        if len(unique_author_labels) > 1:
            min_count = np.min(author_counts)
            AAP += 1 - (min_count / len(cluster_indices))
        else:
            AAP += 1

    ACP /= len(unique_clusters)
    AAP /= len(unique_clusters)

    return ACP, AAP

def calculate_KMetric(ACP, AAP):
    K = ACP / (ACP + AAP)
    return K

def GHAC(mlist,papers,n_clusters=-1):
    paper_weight = np.array(PHNet.loc[papers][papers])
        
    distance=[]
    graph=[]
    
    for i in range(len(mlist)):
        gtmp=[]
        for j in range(len(mlist)):
            if i<j and paper_weight[i][j]!=0:
                cosdis=np.dot(mlist[i],mlist[j])/(np.linalg.norm(mlist[i])*(np.linalg.norm(mlist[j])))                              
                gtmp.append(cosdis*paper_weight[i][j])
            elif i>j:
                gtmp.append(graph[j][i])
            else:
                gtmp.append(0)
        graph.append(gtmp)
    
    distance =np.multiply(graph,-1)
    
    if n_clusters==-1:
        best_m=-10000000
        graph=np.array(graph)
        n_components1, labels = connected_components(graph)
        
        graph[graph<=0.5]=0
        G=nx.from_numpy_matrix(graph)
        
        n_components, labels = connected_components(graph)
        
        for k in range(n_components,n_components1-1,-1):

            
            model_HAC = AgglomerativeClustering(linkage="average",affinity='precomputed',n_clusters=k)
            model_HAC.fit(distance)
            labels = model_HAC.labels_
            
            part= {}
            for j in range (len(labels)):
                part[j]=labels[j]

            mod = community.modularity(part,G)
            if mod>best_m:
                best_m=mod
                best_labels=labels
        labels = best_labels
    else:
        model_HAC = AgglomerativeClustering(linkage="average",affinity='precomputed',n_clusters=n_clusters)
        model_HAC.fit(distance)
        labels = model_HAC.labels_
    
    return labels

def HAC(mlist,papers,n_clusters):
    distance=[]
    for i in range(len(mlist)):
        tmp=[]
        for j in range(len(mlist)):
            if i<j:
                cosdis=np.dot(mlist[i],mlist[j])/(np.linalg.norm(mlist[i])*(np.linalg.norm(mlist[j])))                              
                tmp.append(cosdis)
            elif i>j:
                tmp.append(distance[j][i])
            else:
                tmp.append(0)
        distance.append(tmp)
    
    distance =np.multiply(distance,-1)
    

    model_HAC = AgglomerativeClustering(linkage="average",affinity='precomputed',n_clusters=n_clusters)
    model_HAC.fit(distance)
    labels = model_HAC.labels_ 

    return labels    

def cluster_evaluate(method):
    times = 0
    result = []
    path = 'raw-data_aminer/'
    file_names = os.listdir(path)
    ktrue = []
    kpre = []
    all_pairwise_precision = []
    all_pairwise_recall = []
    all_pairwise_f1 = []
    all_AAP = []
    all_ACP = []
    all_KMetric = []

    for fname in file_names:
        f = open(path + fname, 'r', encoding='utf-8').read()
        text = re.sub(u"&", u" ", f)
        root = ET.fromstring(text)
        correct_labels = []
        papers = []

        mlist = []
        for i in root.findall('publication'):
            correct_labels.append(int(i.find('label').text))
            pid = "i" + i.find('id').text
            mlist.append(pembd[pid])
            papers.append(pid)

        t0 = time.clock()

        if method == "GHAC_nok":  # k is unknown
            labels = GHAC(mlist, papers)
        elif method == "GHAC":  # k is known
            labels = GHAC(mlist, papers, len(set(correct_labels)))
        elif method == "HAC":
            labels = HAC(mlist, papers, len(set(correct_labels)))

        time1 = time.clock() - t0
        times = times + time1

        correct_labels = np.array(correct_labels)
        labels = np.array(labels)

        ktrue.append(len(set(correct_labels)))
        kpre.append(len(set(labels)))

        pairwise_precision, pairwise_recall, pairwise_f1 = pairwise_evaluate(correct_labels, labels)
        ACP, AAP = calculate_ACP_AAP(correct_labels, labels)
        K = calculate_KMetric(ACP, AAP)

        print(fname, "Pairwise Precision:", pairwise_precision)
        print(fname, "Pairwise Recall:", pairwise_recall)
        print(fname, "Pairwise F1:", pairwise_f1)
        print(fname, "ACP:", ACP)
        print(fname, "AAP:", AAP)
        print(fname, "K Metric:", K)

        result.append([fname, pairwise_precision, pairwise_recall, pairwise_f1, ACP, AAP, K])
        all_pairwise_precision.append(pairwise_precision)
        all_pairwise_recall.append(pairwise_recall)
        all_pairwise_f1.append(pairwise_f1)
        all_AAP.append(AAP)
        all_ACP.append(ACP)
        all_KMetric.append(K)


        # Salvar rótulos corretos, rótulos previstos e embeddings em arquivos separados
        save_dir = 'results/cluster_results'
        os.makedirs(save_dir, exist_ok=True)
        save_path_correct = os.path.join(save_dir, fname + '_correct_labels.pkl')
        save_path_predicted = os.path.join(save_dir, fname + '_predicted_labels.pkl')
        save_path_embeddings = os.path.join(save_dir, fname + '_embeddings.pkl')

        with open(save_path_correct, 'wb') as f_correct:
            pickle.dump(correct_labels, f_correct)

        with open(save_path_predicted, 'wb') as f_predicted:
            pickle.dump(labels, f_predicted)

        with open(save_path_embeddings, 'wb') as f_embeddings:
            pickle.dump(mlist, f_embeddings)



    Prec = 0
    Rec = 0
    F1 = 0
    avg_pairwise_precision = np.mean(all_pairwise_precision)
    avg_pairwise_recall = np.mean(all_pairwise_recall)
    avg_pairwise_f1 = np.mean(all_pairwise_f1)
    avg_AAP = np.mean(all_AAP)
    avg_ACP = np.mean(all_ACP)
    avg_KMetric = np.mean(all_KMetric)

    save_csvpath = 'results'

    with open(save_csvpath + method + '.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["name", "Prec", "Rec", "F1", "ACP", "AAP", "K Metric", "Actual", method])
        for i in result:
            Prec = Prec + i[1]
            Rec = Rec + i[2]
            F1 = F1 + i[3]
            writer.writerow(i)

        Prec = Prec / len(result)
        Rec = Rec / len(result)
        F1 = F1 / len(result)
        writer.writerow(["Avg", Prec, Rec, F1, avg_ACP, avg_AAP, avg_KMetric, "0", mean_squared_log_error(ktrue, kpre)])

        print("Cluster method:", method)
        print("Macro-F1")
        print("Precision: ", Prec)
        print("Recall: ", Rec)
        print("F1", F1)
        print("Pairwise Precision Avg:", avg_pairwise_precision)
        print("Pairwise Recall Avg:", avg_pairwise_recall)
        print("Pairwise F1 Avg:", avg_pairwise_f1)
        print("ACP Avg:", avg_ACP)
        print("AAP Avg:", avg_AAP)
        print("K Metric Avg:", avg_KMetric)
        print("avgtime:", times / len(result))
        print("MSLE", mean_squared_log_error(ktrue, kpre))
        print("Accuracy", accuracy_score(ktrue, kpre))

method = 'GHAC_nok'
method = 'GHAC'
# method = 'HAC'

def main():
    cluster_evaluate(method)

if __name__ == "__main__":
    main()