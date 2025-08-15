from transformers import AutoModelForSequenceClassification
from src.config import ModelConfig

def create_model(model_config:ModelConfig, pre_trained:bool = True):

    if pre_trained:
        model = AutoModelForSequenceClassification.from_pretrained(
                            model_config.model_name, 
                    num_labels=model_config.num_classes)
        return model
    else:
        raise NotImplementedError()