import torch
import sys
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Union
from tqdm import tqdm

try:
    from config import NER_ID2LABEL, NER_LABEL2ID, DEVICE, MAX_LENGTH
except ImportError:
    print("NER_PREDICTOR: Ошибка импорта конфигурации. Убедитесь, что config.py доступен.")
    sys.exit(1)

class NERPredictor:
    """
    NER компонент пайплайна.
    """
    def __init__(self, model_name_or_path: str, model_weights_path: str):
        """
        Инициализирует NER компонент пайплайна.

        Args:
            model_name_or_path (str): Имя или путь к базовой модели.
            model_weights_path (str): Путь к файлу с дообученными весами модели.
        """
        self.model_name_or_path = model_name_or_path
        self.model_weights_path = model_weights_path        
        self.id2label: Dict[int, str] = NER_ID2LABEL
        self.label2id: Dict[str, int] = NER_LABEL2ID
        self.num_labels: int = len(self.label2id) if self.label2id else 0        
        self.device = DEVICE
        self.max_length = MAX_LENGTH
        self.tokenizer: Union[AutoTokenizer, None] = None
        self.model: Union[AutoModelForTokenClassification, None] = None        
        self._load_resources()

    def _load_resources(self):
        """
        Загружает токенизатор и модель NER.
        """
        if not self.id2label or not self.label2id:
            print("NERPredictor: id2label или label2id не загружены из конфигурации.")
            return 
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            print("NERPredictor: Токенизатор загружен.")
        except Exception as e:
            print(f"NERPredictor: Ошибка загрузки токенизатора: {e}")
            raise 
        try:
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name_or_path,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id)
            self.model.load_state_dict(torch.load(self.model_weights_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("NERPredictor: Модель загружена и переведена в режим инференса.")
        except Exception as e:
            print(f"NERPredictor: Ошибка загрузки модели: {e}")
            raise 
        
    def _extract_entities_from_bio_tags(self, token_list: List[str], label_list: List[str]) -> List[Dict[str, Union[str, None]]]:
        """
        Извлекает сущности из последовательности токенов и их BIO-меток.
        
        Args:
            token_list (List[str]): Список токенов.
            label_list (List[str]): Список BIO-меток, соответствующий токенам.

        Returns:
            List[Dict[str, Union[str, None]]]: Список словарей, где каждый словарь
                                                представляет извлеченную сущность.
                                                Формат: {"text": "текст сущности", "type": "тип сущности"}
        """
        entities = []
        current_entity_tokens = []
        current_entity_type = None
        for token, label_str in zip(token_list, label_list):
            if label_str.startswith("B-"):
                if current_entity_tokens:
                    entities.append({
                        "text": self.tokenizer.convert_tokens_to_string(current_entity_tokens).strip(),
                        "type": current_entity_type})
                current_entity_tokens = [token]
                current_entity_type = label_str.split('-')[1]
            elif label_str.startswith("I-"):
                tag_type = label_str.split('-')[1]
                if current_entity_tokens and tag_type == current_entity_type:
                    current_entity_tokens.append(token)
                else:
                    if current_entity_tokens:
                        entities.append({
                            "text": self.tokenizer.convert_tokens_to_string(current_entity_tokens).strip(),
                            "type": current_entity_type})
                    current_entity_tokens = []
                    current_entity_type = None
            else: 
                if current_entity_tokens:
                    entities.append({
                        "text": self.tokenizer.convert_tokens_to_string(current_entity_tokens).strip(),
                        "type": current_entity_type})
                current_entity_tokens = []
                current_entity_type = None        
        if current_entity_tokens:
            entities.append({
                "text": self.tokenizer.convert_tokens_to_string(current_entity_tokens).strip(),
                "type": current_entity_type})
        return entities
    
    def predict(self, text_list: List[str], batch_size_inference: int = 16) -> List[List[Dict[str, str]]]:
        """
        Извлекает именованные сущности из списка входных текстов.
        
        Args:
            text_list (List[str]): Список текстов для обработки.
            batch_size_inference (int, optional): Размер батча для инференса.

        Returns:
            List[List[Dict[str, str]]]: Список списков, где каждый внутренний список
                                         содержит словари для найденных персон
                                         в соответствующем входном тексте.
                                         Формат: {"text": "имя персоны", "type": "PER"}.
        """
        if not self.model or not self.tokenizer or not self.id2label:
            print("NERPredictor: Модель, токенизатор или карта меток не загружены. Предсказание невозможно.")
            return [[] for _ in text_list]
        all_results_from_all_batches = []        
        for i in tqdm(range(0, len(text_list), batch_size_inference), desc="NER Batch Inference (Predictor)"):
            batch_texts = text_list[i:i + batch_size_inference]            
            encodings = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt')            
            input_ids_batch = encodings['input_ids'].to(self.device)
            attention_mask_batch = encodings['attention_mask'].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
                logits_batch = outputs.logits
                predictions_batch_ids = torch.argmax(logits_batch, dim=-1)
            for j in range(predictions_batch_ids.shape[0]):
                pred_ids_single = predictions_batch_ids[j].cpu().numpy()
                input_ids_single = input_ids_batch[j].cpu().numpy()
                attention_mask_single = attention_mask_batch[j].cpu().numpy()                
                actual_tokens_single = []
                predicted_token_labels_single = []
                raw_tokens_single = self.tokenizer.convert_ids_to_tokens(input_ids_single)
                for k in range(len(input_ids_single)):
                    if attention_mask_single[k] == 1:
                        if raw_tokens_single[k] not in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                            actual_tokens_single.append(raw_tokens_single[k])
                            predicted_token_labels_single.append(self.id2label.get(pred_ids_single[k], 'O'))                
                entities_in_text = self._extract_entities_from_bio_tags(actual_tokens_single, predicted_token_labels_single)
                person_entities = [entity for entity in entities_in_text if entity.get("type") == "PER"]
                all_results_from_all_batches.append(person_entities)                
        return all_results_from_all_batches