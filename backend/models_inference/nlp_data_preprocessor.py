import json
import os
from typing import List, Dict, Tuple, Any

class NLPDataPreprocessor:
    """
    Класс для чтения загруженных данных из VK и подготовки 
    их для NLP-анализа.
    """
    def __init__(self, max_text_preview_length: int = 150):
        """
        Инициализирует обработчик.

        Args:
            max_text_preview_length (int, optional): Максимальная длина превью текста для сохранения в метаданных. 
        """
        self.max_text_preview_length = max_text_preview_length
    def _extract_post_data(self, record: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Извлекает текст и метаданные из записи о посте.
        
        Args:
            record (Dict[str, Any]): Словарь, представляющий одну запись о посте.

        Returns:
            Union[Tuple[str, Dict[str, Any]], Tuple[None, None]]: 
                Кортеж (post_text, metadata), если текст поста не пустой.
                Кортеж (None, None), если текст поста пустой или отсутствует.
                metadata - словарь с извлеченными метаданными для поста.
        """
        post_text = record.get("text", "").strip()
        if not post_text:
            return None, None        
        metadata = {
            "source_type": "post",
            "source_id": record.get("vk_post_id"),
            "group_id": record.get("vk_group_id"),
            "group_name": record.get("group_name"),
            "date_timestamp": record.get("date"),
            "original_text_preview": post_text[:self.max_text_preview_length]}
        return post_text, metadata

    def _extract_comment_data(self, comment_record: Dict[str, Any], main_post_record: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Извлекает текст и метаданные из записи о комментарии.
        
        Args:
            comment_record (Dict[str, Any]): Словарь, представляющий одну запись о комментарии.
            main_post_record (Dict[str, Any]): Словарь, представляющий запись о родительском посте,
                                               к которому относится комментарий.

        Returns:
            Union[Tuple[str, Dict[str, Any]], Tuple[None, None]]: 
                Кортеж (comment_text, metadata), если текст комментария не пустой.
                Кортеж (None, None), если текст комментария пустой или отсутствует.
                metadata - словарь с извлеченными метаданными для комментария.
        """
        comment_text = comment_record.get("text", "").strip()
        if not comment_text:
            return None, None
        metadata = {
            "source_type": "comment",
            "source_id": comment_record.get("vk_comment_id"),
            "post_id_parent": main_post_record.get("vk_post_id"),
            "group_id": main_post_record.get("vk_group_id"),
            "group_name": main_post_record.get("group_name"),
            "commenter_id": comment_record.get("from_id"),
            "date_timestamp": comment_record.get("date"),
            "original_text_preview": comment_text[:self.max_text_preview_length]}
        return comment_text, metadata

    def extract_and_prepare_input(self, parsed_data_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Читает загруженные данные из JSONL файла, извлекает тексты для NLP анализа
        и собирает соответствующие метаданные.

        Args:
            parsed_data_path (str): Путь к файлу .jsonl с результатами парсинга VK.

        Returns:
            Tuple[List[str], List[Dict[str, Any]]]:
                - texts_for_nlp (List[str]): Список чистых текстов (посты и комментарии).
                - metadata_for_nlp (List[Dict[str, Any]]): Список словарей с метаданными,
                  соответствующий каждому тексту в texts_for_nlp.
        """
        texts_for_nlp: List[str] = []
        metadata_for_nlp: List[Dict[str, Any]] = []
        if not os.path.exists(parsed_data_path):
            print(f"NLPDataPreprocessor: Файл с спарсенными данными не найден: {parsed_data_path}")
            return texts_for_nlp, metadata_for_nlp            
        try:
            with open(parsed_data_path, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, 1):
                    try:
                        record = json.loads(line)                        
                        post_text, post_meta = self._extract_post_data(record)
                        if post_text:
                            texts_for_nlp.append(post_text)
                            metadata_for_nlp.append(post_meta)                        
                        for comment in record.get("comments", []):
                            comment_text, comment_meta = self._extract_comment_data(comment, record)
                            if comment_text:
                                texts_for_nlp.append(comment_text)
                                metadata_for_nlp.append(comment_meta)                                
                    except json.JSONDecodeError:
                        print(f"NLPDataPreprocessor: Ошибка декодирования JSON в строке {line_number} файла {parsed_data_path}")
                    except Exception as e_rec: 
                        post_id_for_log = record.get('vk_post_id', 'N/A') if isinstance(record, dict) else 'N/A'
                        print(f"NLPDataPreprocessor: Ошибка обработки записи (вероятно, post_id: {post_id_for_log}) в строке {line_number}: {e_rec}")
            print(f"NLPDataPreprocessor: Подготовлено {len(texts_for_nlp)} текстов для NLP анализа.")
        except Exception as e_file:
            print(f"NLPDataPreprocessor: Ошибка чтения файла {parsed_data_path}: {e_file}")
            raise RuntimeError(f"Не удалось прочитать файл {parsed_data_path}") from e_file            
        return texts_for_nlp, metadata_for_nlp