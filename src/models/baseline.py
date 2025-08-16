import json
from pathlib import Path
import hashlib
import joblib
import pickle
from typing import List, Dict
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import precision_score,recall_score,f1_score,balanced_accuracy_score,confusion_matrix
from src.data_processing import get_processed_dataset
from src.config import DataConfig, BaselineModelConfig


class LogisticRegressionwithTFIDF(BaseEstimator):

    def __init__(self, config:BaselineModelConfig):

        self.config = config
        self.model = Pipeline([('tfidf', TfidfVectorizer(ngram_range = config.ngram_range,
                                                         min_df = config.min_df,
                                                         max_df = config.max_df,
                                                         strip_accents = config.strip_accents,
                                                         stop_words= config.stop_words,
                                                         token_pattern = config.token_pattern,
                                                         max_features = config.max_features)),

                              ('classifier', LogisticRegression(C=config.C,
                                                            multi_class= config.multi_class,
                                                            class_weight=config.class_weight,
                                                            solver=config.solver,           
                                                            max_iter=config.max_iter,
                                                            random_state=config.random_state))])
        
    def fit(self, inputs, labels):
        return self.model.fit(inputs, labels)
    
    def predict(self, inputs):
        return self.model.predict(inputs)
    
    def predict_proba(self, inputs):
        return self.model.predict_proba(inputs)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)
    
    def set_params(self, **params):
        self.model.set_params(**params)
        return self
    
    def cross_validate(self, X, y, cv_obj,
                        grid_params):
        

        grid_search = GridSearchCV(self.model, grid_params, 
                                   cv= cv_obj,
                                   scoring='f1_macro')
        
        grid_search.fit(X, y)

        return grid_search


def train_cv(data_config: DataConfig, model_config: BaselineModelConfig, cv_param_grid: Dict[str, List]):

        ds_processed = get_processed_dataset(data_config)

        Xtrain = list(ds_processed["train"]["input"])
        ytrain = list(ds_processed["train"]["label"])

        Xval = list(ds_processed["validation"]["input"])
        yval = list(ds_processed["validation"]["label"])

        X_combined = Xtrain + Xval
        y_combined = ytrain + yval


        val_fold = [-1]* len(Xtrain) + [0]*len(Xval)
        cv = PredefinedSplit(val_fold)

        baseline_mdl = LogisticRegressionwithTFIDF(model_config)
        print(f" Running grid search cross-valdiaiton (this may take a while...)")

        grid_search = baseline_mdl.cross_validate(X_combined, y_combined, 
                                                           cv, cv_param_grid)
        
        print(f"Completed cross-valdiation")
        print(f"Best cross valdiation F1 score = {grid_search.best_score_:.4f}")
        print(f"Best cross valdiation model params = {grid_search.best_params_}")

        Xtest = list(ds_processed["test"]["input"])
        ytest = list(ds_processed["test"]["label"])
        ytest_pred = grid_search.best_estimator_.predict(Xtest)

        test_evaluation_metrics = dict(
                                  test_acc_balanced = balanced_accuracy_score(ytest, ytest_pred, adjusted = True),
                                  test_precision_macro = precision_score(ytest,ytest_pred, average = 'macro'),
                                  test_recall_macro = recall_score(ytest,ytest_pred, average = 'macro'),
                                  test_f1_macro = f1_score(ytest,ytest_pred, average = 'macro'),
                                  test_confusion_matrix = confusion_matrix(ytest, ytest_pred))
        
        return grid_search, test_evaluation_metrics


def main(data_config, cv_param_grid):

        cv_version_id = generate_gridsearch_versionid(cv_param_grid)
        cache_dir = f"./models/baseline_{data_config.version_id}_{cv_version_id}"
        cache_path = Path(cache_dir)
        
        if cache_path.exists():

            with open(cache_path / "best_params.json",'r') as f:
                params = json.load(f)
            
            print(f" Loaded cached model with best CV params: {params}")
        
            grid_search = joblib.load(cache_path / "grid_search.pkl")        
            best_model = joblib.load(cache_path / "best_model.pkl")

            with open(cache_path / "test_evals.pkl",'rb') as f:
                test_evaluation_metrics = pickle.load(f)

            print(f" Loaded best model's test performance: {test_evaluation_metrics}")
        
        else:

            print("No cached model found. Training...")      
            
            model_config = BaselineModelConfig()
            grid_search, test_evaluation_metrics = train_cv(data_config, model_config, cv_param_grid)
            best_model = grid_search.best_estimator_
            cache_path.mkdir(parents=True, exist_ok=True)
            joblib.dump(grid_search, cache_path / "grid_search.pkl")
            joblib.dump(best_model, cache_path / "best_model.pkl")
            
            # Log the cross-valdiaiton grid params
            with open(cache_path / "grid_params.json", 'w') as f:
                 json.dump(cv_param_grid, f, indent=2)
        
            # Log the best cross-valdiaiton params resulting from grid search
            with open(cache_path / "best_params.json", 'w') as f:
                json.dump(grid_search.best_params_, f, indent=2)

            with open(cache_path / "test_evals.pkl", 'wb') as f:
                pickle.dump(test_evaluation_metrics, f)

        return grid_search, best_model, test_evaluation_metrics
        

def generate_gridsearch_versionid(grid_params):
    param_str = json.dumps(grid_params, sort_keys = True)
    grid_hash = hashlib.md5(param_str.encode())
    return grid_hash.hexdigest()[:8]

if __name__ == '__main__':

    data_config = DataConfig(num_categories_per_subject=15)    
    cv_param_grid = {
    'tfidf__max_features': [1000, 5000, 10000, 20000],
    'tfidf__ngram_range': [(1,2),(1,3)],
    'classifier__C': [0.1, 1.0, 10.0]}
    grid_search, best_model, test_evaluation_metrics = main(data_config, cv_param_grid)