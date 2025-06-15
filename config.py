# File: config.py
# Description: Модуль конфигурации приложения для объектно-ориентированного 
#              анализа тональности групп ВКонтакте. Содержит пути к данным 
#              и моделям, настройки API, параметры моделей и другие глобальные константы.

import torch
import os
import json

# --- БАЗОВЫЕ ПУТИ И ДИРЕКТОРИИ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODELS_DIR = os.path.join(BASE_DIR, "models_saved")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data_processed")
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
# --- ОБЩИЕ НАСТРОЙКИ МОДЕЛЕЙ ---
DEVICE_TYPE = "cuda" 
DEVICE = torch.device(DEVICE_TYPE if DEVICE_TYPE == "cuda" and torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128  
# --- НАСТРОЙКИ NER МОДЕЛИ ---
NER_MODEL_NAME_OR_PATH = "sberbank-ai/ruRoberta-large" 
NER_MODEL_FILENAME = "best_ner_model_sberbank-ai_ruRoberta-large.bin"
NER_MODEL_WEIGHTS_PATH = os.path.join(MODELS_DIR, NER_MODEL_FILENAME)
NER_LABEL_MAP_FILENAME = "ner_label_maps.json"
NER_LABEL_MAP_PATH = os.path.join(MODELS_DIR, NER_LABEL_MAP_FILENAME)
NER_INFERENCE_BATCH_SIZE = 32
# --- НАСТРОЙКИ TESA МОДЕЛИ ---
TESA_MODEL_NAME_OR_PATH = "sberbank-ai/ruRoberta-large"
TESA_MODEL_FILENAME = "best_tesa_model_sberbank-ai_ruRoberta-large.bin"
TESA_MODEL_WEIGHTS_PATH = os.path.join(MODELS_DIR, TESA_MODEL_FILENAME)
TESA_LABEL_MAP_FILENAME = "tesa_label_maps.json"
TESA_LABEL_MAP_PATH = os.path.join(MODELS_DIR, TESA_LABEL_MAP_FILENAME)
TESA_INFERENCE_BATCH_SIZE = 16
# --- ЗАГРУЗКА СЛОВАРЕЙ МЕТОК ДЛЯ МОДЕЛЕЙ ---
def _load_json_map(path: str, map_name: str = "") -> tuple[dict[int, str], dict[str, int]]:
    """
    Вспомогательная функция для загрузки словарей 'id2label' и 'label2id' 
    из указанного JSON-файла. Ключи 'id2label' конвертируются в int.

    Args:
        path (str): Путь к JSON-файлу.
        map_name (str, optional): Имя словаря для использования в сообщениях об ошибках.
        
    Returns:
        tuple[dict[int, str], dict[str, int]]: Кортеж (id2label_map, label2id_map).
                                               В случае ошибки возвращает пустые словари.
    """
    id2label_map, label2id_map = {}, {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        id2label_map = {int(k): v for k, v in data.get('id2label', {}).items()}
        label2id_map = data.get('label2id', {})
        if not id2label_map or not label2id_map:
            if data.get('id2label') is not None or data.get('label2id') is not None:
                 print(f"CONFIG: '{map_name}' id2label или label2id не найдены корректно или пусты в файле: {path}")
                 pass 
    except FileNotFoundError:
        print(f"CONFIG: Файл словаря меток '{map_name}' не найден по пути: {path}")
        pass 
    except json.JSONDecodeError:
        print(f"CONFIG: Ошибка декодирования JSON в файле '{map_name}': {path}")
        pass
    except Exception as e: 
        print(f"CONFIG: Неизвестная ошибка при загрузке словаря меток '{map_name}' из {path}: {e}")
        pass
    return id2label_map, label2id_map
NER_ID2LABEL, NER_LABEL2ID = _load_json_map(NER_LABEL_MAP_PATH, "NER")
TESA_ID2LABEL, TESA_LABEL2ID = _load_json_map(TESA_LABEL_MAP_PATH, "TESA")
# --- НАСТРОЙКИ ПАРСЕРА VK ---
VK_SERVICE_TOKEN = os.getenv("VK_SERVICE_TOKEN")
if not VK_SERVICE_TOKEN:
    print("CONFIG: Переменная окружения VK_SERVICE_TOKEN не установлена")
VK_API_VERSION = "5.131"
VK_API_BASE_URL = "https://api.vk.com/method/"
DEFAULT_GROUP_IDENTIFIERS = ["zlo43"]
POSTS_CHUNK_SIZE = 100
COMMENTS_CHUNK_SIZE = 100
MAX_COMMENTS_PER_POST_SESSION = None 
CONCURRENT_API_REQUESTS_PER_GROUP_SEMAPHORE = 4
DELAY_AFTER_API_CALL_SECONDS = 0.37
# --- НАСТРОЙКИ ОБРАБОТКИ И ХРАНЕНИЯ ДАННЫХ ---
PARSED_DATA_OUTPUT_FILE = os.path.join(DATA_PROCESSED_DIR, "vk_parsed_data_temp.jsonl")
NLP_RESULTS_OUTPUT_DIR = DATA_PROCESSED_DIR
NLP_RESULTS_FILENAME_TEMPLATE = "nlp_analysis_results_{timestamp}.jsonl"