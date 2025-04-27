# main.py
import os
import time
import numpy as np
import argparse

from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import KFold
import time
import utils

def run_pipeline_for_cd_mode(am, cd_mode,order, G):
    innital_cd_begin_time = time.time()
    print(f'{innital_cd_begin_time} {cd_mode} pre cd begin')
    G = utils.community_detection(G,cd_mode=cd_mode,when='origin')
    innital_cd_end_time = time.time()
    print(f'{innital_cd_end_time} {cd_mode}pre cd end,'+str(innital_cd_end_time-innital_cd_begin_time))
    q_origin, nmi_origin, ari_origin = utils.cal_q_nmi_ari(G,when='origin')
    if order == '1':
        all_pairs = np.array(utils.sample_edges_balanced_with_local_search1(G))
    elif order == '2':
        all_pairs = np.array(utils.sample_edges_balanced_with_local_search2(G))
    

    features_with_ids, labels = utils.extract_pairwise_features_with_common_neighbors(
        G = G,
        all_pairs=all_pairs,
        am = am
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_predictions = {
        'XGBoost': [],
        'DecisionTree': [],
        'RandomForest': [],
        'VotingClassifier_soft': [],
        'VotingClassifier_hard': [],
    }
    for train_index, test_index in kf.split(features_with_ids):
        X_train, X_test = features_with_ids[train_index][:,2:], features_with_ids[test_index]
        y_train, _ = labels[train_index], labels[test_index]

        models = {}
        # 3.1 XGBoost
        xgb_param_grid = {
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [50, 100]
        }
        xgb_clf = xgb.XGBClassifier(random_state=42)
        xgb_grid_search = GridSearchCV(
            estimator=xgb_clf,
            param_grid=xgb_param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        xgb_grid_search.fit(X_train, y_train)
        best_xgb_clf = xgb_grid_search.best_estimator_
        models['XGBoost'] = best_xgb_clf

        # 3.2 DecisionTree
        dt_param_grid = {
            'max_depth': [3, 5],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'criterion': ['gini', 'entropy']
        }
        dt_clf = DecisionTreeClassifier(random_state=42)
        dt_grid_search = GridSearchCV(
            estimator=dt_clf,
            param_grid=dt_param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        dt_grid_search.fit(X_train, y_train)
        best_dt_clf = dt_grid_search.best_estimator_
        models['DecisionTree'] = best_dt_clf

        # 3.3 RandomForest
        rf_param_grid = {
            'n_estimators': [50],
            'max_depth': [5, 10]
        }
        rf_clf = RandomForestClassifier(random_state=42)
        rf_grid_search = GridSearchCV(
            rf_clf,
            rf_param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        rf_grid_search.fit(X_train, y_train)
        best_rf_clf = rf_grid_search.best_estimator_
        models['RandomForest'] = best_rf_clf

        # soft vc
        voting_clf_soft = VotingClassifier(
            estimators=[('xgb', best_xgb_clf), ('dt', best_dt_clf), ('rf', best_rf_clf)],
            voting='soft'
        )
        voting_clf_soft.fit(X_train, y_train)
        models['VotingClassifier_soft'] = voting_clf_soft
        
        
        # hard vc
        voting_clf_hard = VotingClassifier(
            estimators=[('xgb', best_xgb_clf), ('dt', best_dt_clf), ('rf', best_rf_clf)],
            voting='hard'
        )
        voting_clf_hard.fit(X_train, y_train)
        models['VotingClassifier_hard'] = voting_clf_hard

        for model_name, model in models.items():
            node_pair_ids = X_test[:, :2]
            if model_name == 'VotingClassifier_hard':
                predictions = model.predict(X_test[:,2:])
            else:
                predictions = model.predict_proba(X_test[:,2:])[:, 1]
            combined = np.hstack((node_pair_ids, predictions.reshape(-1, 1)))
            fold_predictions[model_name].extend(combined)
    def evaluate_model_and_repartition(clf,fold_predictions, model_name):
        G_weighted = utils.adjust_weight(G,cd_mode, fold_predictions[model_name])
        q_new, nmi_new, ari_new = utils.cal_q_nmi_ari(G_weighted, when='weighted')
        compare_str = utils.compare_q_return(q_origin, q_new,nmi_origin,nmi_new,ari_origin,ari_new)
        result_text = []
        result_text.append(f"\n--- {model_name} ---")
        result_text.append(f"Best Params: {clf}")
        result_text.append(f"Weighted Modularity: {q_new}")
        result_text.append(f"Weighted NMI: {nmi_new}")
        result_text.append(f"Weighted ARI: {ari_new}")
        result_text.append(compare_str)
        return "\n".join(result_text)


    result_lines = []
    result_lines.append(f"===== Community Detection Mode: {cd_mode} =====")
    result_lines.append(f"Original Modularity: {q_origin}")
    result_lines.append(f"Original NMI: {nmi_origin}")
    result_lines.append(f"Original ARI: {ari_origin}")
    result_lines.append(f"Feature Matrix Shape: {features_with_ids[:,:2].shape}, Labels Shape: {labels.shape}")

    try:
        result_lines.append(evaluate_model_and_repartition(best_xgb_clf,fold_predictions, "XGBoost"))
    except:
        pass
    try:
        result_lines.append(evaluate_model_and_repartition(best_dt_clf,fold_predictions, "DecisionTree"))
    except:
        pass
    try:
        result_lines.append(evaluate_model_and_repartition(best_rf_clf,fold_predictions, "RandomForest"))
    except:
        pass
    try:
        result_lines.append(evaluate_model_and_repartition(voting_clf_soft, fold_predictions,"VotingClassifier_soft"))
    except:
        pass
    try:
        result_lines.append(evaluate_model_and_repartition(voting_clf_hard, fold_predictions,"VotingClassifier_hard"))
    except:
        pass

    final_str = "\n".join(result_lines)
    return final_str


if __name__ == "__main__":
    os.environ['JOBLIB_TEMP_FOLDER'] = r'C:\Temp'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", required=True, help="Name of the dataset folder, e.g., 'com-dblp'")
    parser.add_argument("--am", required=True, help="algorithm model")
    parser.add_argument("--order", required=True, help="order of the neighbor")
    args = parser.parse_args()

    file_name = args.filename
    am = args.am
    order = args.order
    input_dir = f'./norm_dataset/{file_name}/'
    node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')

    time_str = time.strftime("%Y%m%d_%H%M%S")
    output_file_path = f"results/results_{file_name}_{am}_order_5kf_{time_str}.txt"
    os.makedirs('results', exist_ok=True)
    
    cd_modes = ['louvain', 'infomap', 'leiden', 'fastgreedy']
    
    G = utils.load_graph_with_attributes(node_file_path, edge_file_path)    
    with open(output_file_path, "w", encoding='utf-8') as f:
        for cd_mode in cd_modes:
            begin_time = time.time()
            result_text = run_pipeline_for_cd_mode(am, cd_mode, order, G)
            f.write(result_text)
            f.write("\n\n" + "="*70 + "\n\n")
            print(f"[INFO] Finished cd_mode = {cd_mode}. See {output_file_path} for details.")
            end_time = time.time()
            print(end_time-begin_time)

    print(f"Finished,results saved in: {output_file_path}")
