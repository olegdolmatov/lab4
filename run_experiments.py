import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from grader_contracts.ensembles import EnsembleInput
from ensemble_models import calculate_metrics, prepare_model, run_ensemble_experiment

SEED = 42
MODELS = ["Random Forest", "Extra Trees", "Gradient Boosting", "Random Forest (leaf=1)"]


def main():
    root = Path(__file__).resolve().parent
    df = pd.read_csv(root / "german.csv", sep=";")
    x = df.drop(columns="Creditability")
    y = df["Creditability"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED, stratify=y)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    experiments = []
    for name in MODELS:
        model = prepare_model(name, x_train, SEED)
        cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        model.fit(x_train, y_train)
        positive_class = list(model.named_steps["classifier"].classes_).index(1)
        probabilities = model.predict_proba(x_test)[:, positive_class]
        result = {"model": name, "cv_roc_auc_mean": round(float(cv_scores.mean()), 5), "cv_roc_auc_std": round(float(cv_scores.std()), 5)}
        result.update({k: round(v, 5) for k, v in calculate_metrics(y_test, probabilities).items()})
        experiments.append(result)
        print(result)
    chosen = max(experiments, key=lambda row: row["cv_roc_auc_mean"])["model"]
    print("Лучшая по CV:", chosen)
    contract = run_ensemble_experiment(EnsembleInput(x_train, y_train, x_test, y_test, SEED))
    print("Контрактная функция:", contract.model_name, contract.roc_auc)
    data = {"seed": SEED, "split": "80/20 stratified", "cv": "3-fold StratifiedKFold on train", "selection_metric": "cv_roc_auc_mean", "best_by_cv": chosen, "contract_model": contract.model_name, "experiments": experiments}
    (root / "experiment_results.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
