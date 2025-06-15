import asyncio
import datetime
import json
import os
import sys 
import copy
from typing import List, Dict, Any, Union, Tuple 

try:
    import config 
    from backend.vk_parser import parser as vk_parser_module
    from backend.models_inference.pipeline_runner import create_sentiment_pipeline, SentimentPipeline 
    from backend.models_inference.nlp_data_preprocessor import NLPDataPreprocessor
except ImportError as e:
    print(f"APP_LOGIC: Ошибка импорта: {e}")
    sys.exit(1) 

class AnalysisService:
    """
    Полный цикл объектно-ориентированного анализа 
    тональности текстовых данных из групп ВКонтакте.
    """
    def __init__(self):
        """
        Инициализирует сервис анализа.
        """
        try:
            self.data_preprocessor = NLPDataPreprocessor(max_text_preview_length=getattr(config, 'MAX_TEXT_PREVIEW_LENGTH', 300) )
        except Exception as e_prep:
            print(f"AnalysisService: Ошибка инициализации NLPDataPreprocessor: {e_prep}")
            raise RuntimeError("Ошибка инициализации препроцессора данных") from e_prep
        try:
            self.nlp_pipeline: SentimentPipeline = create_sentiment_pipeline()
            print("AnalysisService: SentimentPipeline успешно создан и модели загружены.")
        except Exception as e_pipeline: 
            print(f"AnalysisService: Ошибка инициализации SentimentPipeline: {e_pipeline}")
            self.pipeline_initialization_error = True 
            raise RuntimeError("Ошибка инициализации NLP пайплайна") from e_pipeline
        else:
            self.pipeline_initialization_error = False
        if hasattr(config, 'TESA_ID2LABEL') and config.TESA_ID2LABEL:
            self.tesa_id2label: Dict[int, str] = config.TESA_ID2LABEL
            print(f"AnalysisService: Карта меток TESA загружена из config ({len(self.tesa_id2label)} меток).")
        else:
            print("AnalysisService: Ошибка: Словарь меток TESA_ID2LABEL не найден.")
            raise ValueError("Карта меток TESA не сконфигурирована.")

    def _aggregate_nlp_results(self, 
                               nlp_results_per_text: List[List[Dict[str, str]]], 
                               metadata_for_nlp: List[Dict[str, Any]]) -> Dict[str, Any]:    
        """
        Агрегирует результаты NLP-анализа.

        Args:
            nlp_results_per_text (List[List[Dict[str, str]]]): Список результатов,
                полученных от NLP пайплайна. Каждый элемент списка соответствует
                одному обработанному тексту и содержит список словарей, где каждый
                словарь имеет ключи "entity" (нормализованное имя),
                "entity_original" (оригинальное имя) и "polarity" (метка тональности).
            metadata_for_nlp (List[Dict[str, Any]]): Список словарей с метаданными,
                соответствующий каждому тексту в 'nlp_results_per_text'. Используется
                для привязки результатов к исходным данным.

         Returns:
            Dict[str, Any]: Словарь, содержащий два ключа:
                - "summary_by_entity_date": Агрегированные данные, где ключи - это
                  нормализованные имена сущностей. Значения - словари, где ключи -
                  это даты (в формате "YYYY-MM-DD"), а значения - словари с
                  подсчитанным количеством упоминаний для каждой тональности.
                - "detailed_mentions": Список словарей. Каждый словарь представляет
                  одно упоминание и содержит "entity_normalized", "entity_original",
                  "polarity", а также информацию из metadata_for_nlp.
        """
        final_aggregated_data: Dict[str, Dict[str, Dict[str, int]]] = {}
        detailed_mentions: List[Dict[str, Any]] = [] 
        if len(nlp_results_per_text) != len(metadata_for_nlp):
            print("AnalysisService: Несовпадение длин результатов NLP и метаданных. Агрегация может быть неполной.")
        for i, text_opinions in enumerate(nlp_results_per_text):
            if i >= len(metadata_for_nlp): 
                print(f"AnalysisService: Нет метаданных для текста с индексом {i}, пропуск.")
                continue             
            meta = metadata_for_nlp[i]
            date_ts = meta.get("date_timestamp")
            if date_ts is None: 
                print(f"AnalysisService: Отсутствует 'date_timestamp' в метаданных для текста {i}, пропуск.")
                continue            
            try: 
                date_key = datetime.datetime.fromtimestamp(date_ts).strftime("%Y-%m-%d")
            except (TypeError, ValueError) as e_date: 
                print(f"AnalysisService: Некорректный timestamp '{date_ts}' в метаданных для текста {i}: {e_date}. Пропуск записи.")
                continue
            for opinion_pair in text_opinions:
                entity_normalized = opinion_pair.get("entity")
                entity_original = opinion_pair.get("entity_original")
                polarity = opinion_pair.get("polarity")

                if not entity_normalized or not polarity: 
                    continue 

                if entity_normalized not in final_aggregated_data:
                    final_aggregated_data[entity_normalized] = {}
                
                if date_key not in final_aggregated_data[entity_normalized]:
                    final_aggregated_data[entity_normalized][date_key] = {
                        label_name: 0 for label_name in self.tesa_id2label.values()
                    }
                    final_aggregated_data[entity_normalized][date_key]["UNKNOWN"] = 0 

                if polarity in final_aggregated_data[entity_normalized][date_key]:
                    final_aggregated_data[entity_normalized][date_key][polarity] += 1
                else:
                    final_aggregated_data[entity_normalized][date_key]["UNKNOWN"] += 1
                
                detailed_mentions.append({
                    "entity_normalized": entity_normalized, 
                    "entity_original": entity_original,     
                    "polarity": polarity,
                    "date": date_key,
                    "timestamp": date_ts,
                    "source_type": meta.get("source_type"),
                    "source_id": meta.get("source_id"),
                    "post_id_if_comment": meta.get("post_id_parent"), 
                    "group_name": meta.get("group_name"),
                    "text_preview": meta.get("original_text_preview") 
                })
        
        print(f"AnalysisService: Агрегация завершена. Уникальных (нормализованных) сущностей: {len(final_aggregated_data)}, всего упоминаний: {len(detailed_mentions)}.")
        return {"summary_by_entity_date": final_aggregated_data, "detailed_mentions": detailed_mentions}

    def reaggregate_with_aliases(self, 
                                 current_summary: Dict[str, Dict[str, Dict[str, int]]], 
                                 current_detailed_mentions: List[Dict[str, Any]], 
                                 aliases_to_merge: List[str], 
                                 canonical_name: str) -> Tuple[Dict[str, Dict[str, Dict[str, int]]], List[Dict[str, Any]]]:
        """
        Объединяет данные для нескольких сущностей под одним каноническим именем.
        
        Args:
            current_summary (Dict[str, Dict[str, Dict[str, int]]]): Текущая агрегированная сводка
                в формате {сущность: {дата: {тональность: счетчик}}}.
            current_detailed_mentions (List[Dict[str, Any]]): Текущий список словарей
                с детализированными упоминаниями.
            aliases_to_merge (List[str]): Список строковых имен сущностей,
                которые должны быть объединены.
            canonical_name (str): Строковое каноническое имя, под которым
                будут агрегированы данные алиасов.

        Returns:
            Tuple[Dict[str, Dict[str, Dict[str, int]]], List[Dict[str, Any]]]:
                Кортеж, содержащий обновленные 'new_summary' и 'new_detailed_mentions'.
        """
        print(f"AnalysisService: Переагрегация с объединением {aliases_to_merge} в '{canonical_name}'...")
        if not canonical_name.strip():
            print("AnalysisService ERROR: Каноническое имя не может быть пустым для переагрегации.")
            return current_summary, current_detailed_mentions
        new_summary = copy.deepcopy(current_summary)
        new_detailed_mentions = copy.deepcopy(current_detailed_mentions)
        if canonical_name not in new_summary:
            new_summary[canonical_name] = {}
        for alias in aliases_to_merge:
            if alias == canonical_name:
                continue
            if alias in new_summary:
                for date_key, sentiment_counts in new_summary[alias].items():
                    if date_key not in new_summary[canonical_name]:
                        new_summary[canonical_name][date_key] = {
                            label: 0 for label in self.tesa_id2label.values()}
                        new_summary[canonical_name][date_key]["UNKNOWN"] = 0                    
                    for sentiment, count in sentiment_counts.items():
                        new_summary[canonical_name][date_key][sentiment] = \
                            new_summary[canonical_name][date_key].get(sentiment, 0) + count                
                del new_summary[alias]
            else:
                print(f"AnalysisService WARNING: Объединенная сущность '{alias}' не найдена в текущей сводке для объединения.")
        for mention in new_detailed_mentions:
            if mention.get("entity") in aliases_to_merge:
                mention["entity"] = canonical_name         
        return new_summary, new_detailed_mentions

    async def run_full_analysis(self, group_identifiers_str: str, 
                                date_start_str: str, date_end_str: str
                               ) -> Dict[str, Any]:
        """
        Выполняет полный цикл анализа тональности для указанных групп VK.

        Args:
            group_identifiers_str (str): Строка, содержащая ID или короткие имена
                                         групп ВКонтакте, разделенные запятыми.
            date_start_str (str): Начальная дата периода анализа в формате.
            date_end_str (str): Конечная дата периода анализа в формате.

        Returns:
            Dict[str, Any]: Словарь с результатами анализа. Содержит ключи:
                - "summary" (Dict | None): Агрегированная сводка по сущностям, датам и тональностям.
                                         Формат: {сущность: {дата: {тональность: счетчик}}}.
                - "detailed_results_file" (str | None): Путь к файлу JSONL с детализированными
                                                        упоминаниями. None, если файл не был создан.
                - "message" (str, optional): Информационное сообщение о статусе выполнения.
                - "error" (str, optional): Сообщение об ошибке, если в процессе анализа
                                         произошла проблема, не позволившая его завершить.
        """
        if self.pipeline_initialization_error:
            return {"error": "Критическая ошибка: NLP модели (пайплайн) не были инициализированы."}        
        try:
            group_list = [gid.strip() for gid in group_identifiers_str.split(',') if gid.strip()]
            if not group_list: return {"error": "Список идентификаторов групп не может быть пустым."}            
            start_date_obj = datetime.datetime.strptime(date_start_str, "%Y-%m-%d").date()
            end_date_obj = datetime.datetime.strptime(date_end_str, "%Y-%m-%d").date()
            if start_date_obj > end_date_obj: return {"error": "Начальная дата не может быть позже конечной."}
        except ValueError as e:
            return {"error": f"Неверный формат входных параметров: {e}"}        
        parsed_data_file_path = None
        try:
            print(f"AnalysisService: Запуск парсинга для групп: {group_list}, период: {start_date_obj} - {end_date_obj}")
            parsed_data_file_path = await vk_parser_module.fetch_vk_data(group_list, start_date_obj, end_date_obj)
            print(f"AnalysisService: Данные VK сохранены в: {parsed_data_file_path}")
        except Exception as e_parse: 
            print(f"AnalysisService CRITICAL: Ошибка парсинга данных VK: {e_parse}")
            return {"error": f"Ошибка парсинга данных VK: {str(e_parse)}"}
        if not parsed_data_file_path or not os.path.exists(parsed_data_file_path):
            return {"error": "Файл с результатами парсинга не найден или парсинг не вернул данных."}        
        try:
            texts_for_nlp, metadata_for_nlp = self.data_preprocessor.extract_and_prepare_input(parsed_data_file_path)
        except Exception as e_prep_extract: 
            print(f"AnalysisService CRITICAL: Ошибка предобработки данных для NLP: {e_prep_extract}")
            return {"error": f"Ошибка предобработки данных для NLP: {str(e_prep_extract)}"}
        if not texts_for_nlp:
            return {"message": "Парсинг завершен, но не найдено текстов (постов/комментариев) для NLP анализа."}            
        nlp_results_per_text: List[List[Dict[str, str]]] = []
        try:
            print(f"AnalysisService: Запуск NLP пайплайна для {len(texts_for_nlp)} текстов...")
            nlp_results_per_text = self.nlp_pipeline.run(texts_for_nlp) 
            print("AnalysisService: NLP пайплайн завершен.")
        except Exception as e_nlp: 
            print(f"AnalysisService CRITICAL: Ошибка NLP анализа: {e_nlp}")
            return {"error": f"Ошибка NLP анализа: {str(e_nlp)}"}
        aggregated_and_detailed_results = self._aggregate_nlp_results(nlp_results_per_text, metadata_for_nlp)        
        detailed_mentions_to_save = aggregated_and_detailed_results.get("detailed_mentions", [])
        nlp_output_file_path: Union[str, None] = None
        if detailed_mentions_to_save:
            try:
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = config.NLP_RESULTS_FILENAME_TEMPLATE.format(timestamp=timestamp_str)
                nlp_output_file_path = os.path.join(config.NLP_RESULTS_OUTPUT_DIR, filename)                
                with open(nlp_output_file_path, 'w', encoding='utf-8') as f_out_nlp:
                    for mention_record in detailed_mentions_to_save:
                        f_out_nlp.write(json.dumps(mention_record, ensure_ascii=False) + '\n')
                print(f"AnalysisService: Детальные результаты NLP ({len(detailed_mentions_to_save)} упоминаний) сохранены в: {nlp_output_file_path}")
            except Exception as e_save: 
                print(f"AnalysisService ERROR: Ошибка сохранения детальных результатов NLP: {e_save}")
                nlp_output_file_path = None 
        else:
            print("AnalysisService: Нет детализированных упоминаний для сохранения.")            
        return {
            "summary": aggregated_and_detailed_results.get("summary_by_entity_date"),
            "detailed_results_file": nlp_output_file_path,
            "message": "Анализ успешно завершен."}