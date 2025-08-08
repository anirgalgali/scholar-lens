from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.data_processing import process_dataset
from sklearn.utils import shuffle
from sklearn.metrics import precision_score,recall_score,f1_score,balanced_accuracy_score

class LogisticRegressionwithTFIDF:

    def __init__(self, **model_config):

        self.model = Pipeline([('tfidf', TfidfVectorizer(ngram_range = model_config['ngram_range'],
                                                         min_df =  model_config['min_df'],
                                                         max_df =  model_config['max_df'],
                                                         strip_accents = 'unicode',
                                                         stop_words= 'english',
                                                         token_pattern = model_config['token_pattern'],
                                                         max_features = model_config['max_features']
                                                         )), 
                              ('logistic', LogisticRegression(C=model_config['C'],
                                                            multi_class='multinomial',
                                                            class_weight='balanced',
                                                            solver='lbfgs',           
                                                            max_iter=model_config['max_iter'],
                                                            random_state=model_config['seed']))])
        
    def fit(self, inputs, labels):
        return self.model.fit(inputs, labels)
    
    def predict(self, inputs):
        return self.model.predict(inputs)
    
    def predict_proba(self, inputs):
        return self.model.predict_proba(inputs)
    
def train_baseline_model(dataset_identifier, preprocess_config, model_config):

        ds_filtered, category_label_mapping = process_dataset(dataset_identifier,
                                                               preprocess_config, load_from_cache = True)
        
        print(f"=========CATEGORY-TO-LABEL-MAPPING=============")
        print(category_label_mapping)
        print(f"-----------------------------------------------")
        
        Xtrain = list(ds_filtered["train"]["input"])
        ytrain = list(ds_filtered["train"]["label"])

        Xval = list(ds_filtered["validation"]["input"])
        yval = list(ds_filtered["validation"]["label"])

        baseline_mdl = LogisticRegressionwithTFIDF(**model_config)
        _ = baseline_mdl.fit(Xtrain, ytrain)
        predicted_val_labels = baseline_mdl.predict(Xval)

        print(f"Balanced Accuracy: {balanced_accuracy_score(yval, predicted_val_labels,adjusted = True)}")
        print(f"Precsion Accuracy  (macro): {precision_score(yval, predicted_val_labels,average = 'macro')}")
        print(f"Recall  (macro): {recall_score(yval, predicted_val_labels,average = 'macro')}")
        print(f"F1 score (macro): {f1_score(yval, predicted_val_labels,average = 'macro')}")

        return predicted_val_labels, baseline_mdl

if __name__ == '__main__':
    
    preprocess_config = dict(subjects = ['Physics', 'Mathematics', 'Computer Science'], top_k = 8)
    
    model_config = dict(ngram_range = (1,2),
                        min_df = 2,
                        max_df = 0.95,
                        token_pattern = r'\b[\w\-]+\b',
                        max_features = 20000,
                        C = 1.0,
                        max_iter = 1000,
                        seed = 42)
    
    predictions, fitted_mdl = train_baseline_model("TimSchopf/arxiv_categories", preprocess_config, model_config)