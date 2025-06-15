import sys
from typing import List, Dict, Tuple, Any 
from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger

try:
    import config 
    from backend.models_inference.ner_predictor import NERPredictor
    from backend.models_inference.tesa_predictor import TESAPredictor
    print("PIPELINE_RUNNER: Модули config, NERPredictor, TESAPredictor успешно импортированы.")
except ImportError as e:
    print(f"PIPELINE_RUNNER: Ошибка импорта: {e}")
    sys.exit(1)

class SentimentPipeline:
    """
    Полный процесс анализа тональности.
    """
    def __init__(self, ner_predictor: NERPredictor, tesa_predictor: TESAPredictor, 
                 ner_batch_size: int, tesa_batch_size: int):
        """
        Инициализирует пайплайн анализа тональности.

        Args:
            ner_predictor (NERPredictor): Экземпляр предиктора NER.
            tesa_predictor (TESAPredictor): Экземпляр предиктора TESA.
            ner_batch_size (int): Размер батча для NER инференса.
            tesa_batch_size (int): Размер батча для TESA инференса.
        """
        self.ner_predictor = ner_predictor
        self.tesa_predictor = tesa_predictor
        self.ner_batch_size = ner_batch_size
        self.tesa_batch_size = tesa_batch_size
        
        try:
            self.segmenter = Segmenter()
            self.morph_vocab = MorphVocab()
            emb = NewsEmbedding() 
            self.morph_tagger = NewsMorphTagger(emb) 
            self.lemmatization_enabled = True
        except Exception as e_natasha:
            print(f"SentimentPipeline: Не удалось инициализировать компоненты Natasha: {e_natasha}")
            raise RuntimeError("Ошибка инициализации морфологического анализатора") from e_natasha

    def _lemmatize_name(self, name_phrase: str) -> str:
        """
        Нормализует имя персоны: приводит слова к начальной форме
        и к единому регистру.
        Возвращает пустую строку, если входная строка пуста или состоит только из пробелов.
        """
        if not name_phrase or not name_phrase.strip():
            return ""
        
        doc = Doc(name_phrase)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        normalized_words = []
        for token in doc.tokens:        
            is_initial = (len(token.text) == 2 and token.text[1] == '.' and token.text[0].isalpha()) or \
                         (len(token.text) == 1 and token.text[0].isupper() and token.text.isalpha()) 

            if is_initial:
                if len(token.text) == 2 and token.text[1] == '.':
                    normalized_words.append(token.text[0].upper() + ".")
                else: 
                    normalized_words.append(token.text.upper())
            elif token.lemma:
                normalized_words.append(token.lemma.capitalize())
            else:
                normalized_words.append(token.text.capitalize())
                
        return " ".join(normalized_words)
    
    def run(self, text_list: List[str]) -> List[List[Dict[str, str]]]:
        """
        Запускает полный анализ тональности для списка текстов.

        Args:
            text_list (List[str]): Список текстов для анализа.

        Returns:
            List[List[Dict[str, str]]]: Список списков, где каждый внутренний список
                                         содержит словари для каждой найденной сущности
                                         в соответствующем тексте и ее тональности.
                                         Формат словаря: {"entity": "нормализованное_имя_персоны", "entity_original": "оригинальное_имя_персоны_из_текста", "polarity": "метка_тональности"}
        """
        if not text_list:
            print("SentimentPipeline: Возвращен пустой результат.")
            return []        
        print(f"SentimentPipeline: Вызов NERPredictor.predict с batch_size={self.ner_batch_size}")
        try:
            all_ner_results_per_text = self.ner_predictor.predict(text_list, batch_size_inference=self.ner_batch_size)
        except Exception as e_ner:
            print(f"SentimentPipeline: Ошибка на этапе NER: {e_ner}")
            return [[] for _ in text_list]         
        print(f"SentimentPipeline: NER этап завершен. Получено результатов для {len(all_ner_results_per_text)} текстов.")
        all_tesa_input_pairs: List[Tuple[str, str]] = []
        tesa_input_map_back: List[Dict[str, Any]] = []
        for text_idx, (original_text, ner_entities_for_text) in enumerate(zip(text_list, all_ner_results_per_text)):
            if ner_entities_for_text: 
                for entity_info in ner_entities_for_text:
                    entity_text = entity_info.get("text")
                    if entity_text and isinstance(entity_text, str):
                        all_tesa_input_pairs.append((original_text, entity_text))
                        tesa_input_map_back.append({"text_idx": text_idx, "entity_text": entity_text})
        
        final_pipeline_results: List[List[Dict[str, str]]] = [[] for _ in text_list]
        if all_tesa_input_pairs:
            print(f"SentimentPipeline: TESA: Подготовлено {len(all_tesa_input_pairs)} пар (текст, сущность) для анализа тональности.")
            print(f"SentimentPipeline: Вызов TESAPredictor.predict с batch_size={self.tesa_batch_size}")
            try:
                predicted_polarities = self.tesa_predictor.predict(all_tesa_input_pairs, batch_size_inference=self.tesa_batch_size)
            except Exception as e_tesa:
                print(f"SentimentPipeline: Ошибка на этапе TESA: {e_tesa}")
                predicted_polarities = ["ERROR_TESA_PIPELINE"] * len(all_tesa_input_pairs)
            if len(predicted_polarities) == len(tesa_input_map_back):
                for i, map_info in enumerate(tesa_input_map_back):
                    original_text_index = map_info["text_idx"]
                    entity_text = map_info["entity_text"]
                    polarity = predicted_polarities[i]
                    normalized_entity_text = self._lemmatize_name(entity_text)
                    
                    if not normalized_entity_text:
                        normalized_entity_text = entity_text.capitalize() 

                    final_pipeline_results[original_text_index].append({
                        "entity": normalized_entity_text,        
                        "entity_original": entity_text,      
                        "polarity": polarity
                    })                    
            else:
                print("SentimentPipeline: Несовпадение длин результатов TESA и словаря сущностей.")
                for map_info in tesa_input_map_back:
                    original_text_index = map_info["text_idx"]
                    entity_text = map_info["entity_text"]
                    final_pipeline_results[original_text_index].append({
                        "entity": entity_text,
                        "polarity": "ERROR_TESA_LENGTH_MISMATCH"})
        else:
            print("SentimentPipeline: TESA: Нет сущностей, извлеченных NER, для анализа тональности.")        
        return final_pipeline_results

def create_sentiment_pipeline() -> SentimentPipeline:
    """
    Создает и возвращает экземпляр SentimentPipeline с загруженными компонентами.
    """
    try:
        ner_predictor_instance = NERPredictor(
            model_name_or_path=config.NER_MODEL_NAME_OR_PATH,
            model_weights_path=config.NER_MODEL_WEIGHTS_PATH)
        tesa_predictor_instance = TESAPredictor(
            model_name_or_path=config.TESA_MODEL_NAME_OR_PATH,
            model_weights_path=config.TESA_MODEL_WEIGHTS_PATH)
    except Exception as e:
        print(f"PIPELINE_RUNNER: Ошибка при создании экземпляров предикторов: {e}")
        raise RuntimeError(f"Не удалось инициализировать предикторы: {e}") from e
    pipeline_instance = SentimentPipeline(
        ner_predictor=ner_predictor_instance,
        tesa_predictor=tesa_predictor_instance,
        ner_batch_size=config.NER_INFERENCE_BATCH_SIZE,
        tesa_batch_size=config.TESA_INFERENCE_BATCH_SIZE)    
    print("PIPELINE_RUNNER: SentimentPipeline успешно создан и готов к работе.")
    return pipeline_instance