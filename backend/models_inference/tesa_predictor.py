import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Union, Dict
from tqdm import tqdm
import sys

try:
    from config import TESA_ID2LABEL, TESA_LABEL2ID, DEVICE, MAX_LENGTH
except ImportError:
    print("TESA_PREDICTOR: Ошибка импорта конфигурации. Убедитесь, что config.py доступен.")
    sys.exit(1)

class TESAPredictor:
    """
    TESA компонент пайплайна.
    """
    def __init__(self, model_name_or_path: str, model_weights_path: str):
        """
        Инициализирует TESA компонент.

        Args:
            model_name_or_path (str): Имя или путь к базовой модели/архитектуре.
            model_weights_path (str): Путь к файлу с дообученными весами модели.
        """
        self.model_name_or_path = model_name_or_path
        self.model_weights_path = model_weights_path
        self.id2label: Dict[int, str] = TESA_ID2LABEL
        self.label2id: Dict[str, int] = TESA_LABEL2ID
        self.num_labels: int = len(self.label2id) if self.label2id else 0
        self.device = DEVICE
        self.max_length = MAX_LENGTH        
        self.tokenizer: Union[AutoTokenizer, None] = None
        self.model: Union[AutoModelForSequenceClassification, None] = None        
        self._load_resources()

    def _load_resources(self):
        """
        Загружает токенизатор и модель TESA.
        Словари меток уже загружены из config.py.
        """
        if not self.id2label or not self.label2id:
            print("ESAPredictor: id2label или label2id не загружены из конфигурации.")
            return        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            print("TESAPredictor: Токенизатор загружен.")
        except Exception as e:
            print(f"TESAPredictor: Ошибка загрузки токенизатора: {e}")
            raise
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id)
            self.model.load_state_dict(torch.load(self.model_weights_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("TESAPredictor: Модель загружена и переведена в режим инференса.")
        except Exception as e:
            print(f"TESAPredictor: Ошибка загрузки модели: {e}")
            raise
    def predict(self, sentence_entity_pairs: List[Tuple[str, str]], batch_size_inference: int = 16) -> List[str]:
        """
        Определяет тональность для списка пар (предложение, текст_сущности).
        
        Args:
            sentence_entity_pairs (List[Tuple[str, str]]): Список пар для обработки.
            batch_size_inference (int, optional): Размер батча для инференса.

        Returns:
            List[str]: Список строковых меток тональности для каждой входной пары.
        """
        if not self.model or not self.tokenizer or not self.id2label:
            print("TESAPredictor: Модель, токенизатор или карта меток не загружены. Предсказание невозможно.")
            return ["ERROR_PREDICTOR_NOT_LOADED"] * len(sentence_entity_pairs)
        all_polarities_from_all_batches = []
        for i in tqdm(range(0, len(sentence_entity_pairs), batch_size_inference), desc="TESA Batch Inference (Predictor)"):
            batch_pairs = sentence_entity_pairs[i:i + batch_size_inference]            
            batch_sentences = [pair[0] for pair in batch_pairs]
            batch_entities = [pair[1] for pair in batch_pairs]            
            valid_indices = []
            filtered_sentences = []
            filtered_entities = []
            for idx, (sent, ent) in enumerate(zip(batch_sentences, batch_entities)):
                if sent and ent and isinstance(sent, str) and isinstance(ent, str):
                    valid_indices.append(idx)
                    filtered_sentences.append(sent)
                    filtered_entities.append(ent)            
            if not filtered_sentences:
                all_polarities_from_all_batches.extend(["INVALID_INPUT"] * len(batch_pairs))
                continue
            encodings = self.tokenizer(
                filtered_sentences,
                filtered_entities,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt')            
            input_ids_batch = encodings['input_ids'].to(self.device)
            attention_mask_batch = encodings['attention_mask'].to(self.device)
            token_type_ids_batch = encodings.get('token_type_ids')
            if token_type_ids_batch is not None:
                token_type_ids_batch = token_type_ids_batch.to(self.device)
            with torch.no_grad():
                model_inputs = {'input_ids': input_ids_batch, 'attention_mask': attention_mask_batch}
                if token_type_ids_batch is not None:
                    model_inputs['token_type_ids'] = token_type_ids_batch                
                outputs = self.model(**model_inputs)
                logits_batch = outputs.logits
                predictions_batch_ids = torch.argmax(logits_batch, dim=-1)
            batch_polarities_temp = ["UNKNOWN_SENTIMENT"] * len(batch_pairs)
            for original_idx_in_batch, valid_pred_id in zip(valid_indices, predictions_batch_ids.cpu().numpy()):
                batch_polarities_temp[original_idx_in_batch] = self.id2label.get(valid_pred_id.item(), "UNKNOWN_SENTIMENT_ID")            
            all_polarities_from_all_batches.extend(batch_polarities_temp)                
        return all_polarities_from_all_batches