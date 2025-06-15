import streamlit as st
import pandas as pd
import datetime
import asyncio
import sys
import os
import json
import copy
from collections import Counter
from typing import List, Dict, Any, Union, Tuple

CURRENT_SCRIPT_DIR_UI = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_UI = os.path.dirname(CURRENT_SCRIPT_DIR_UI)
if PROJECT_ROOT_UI not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_UI)

try:
    import config as config_module
    from backend.app_logic import AnalysisService 
    from backend.vk_parser.parser import AsyncVKAPI 
    import aiohttp 
    if 'ui_module_loaded_print_once' not in st.session_state:
        print("UI: Основные модули бэкенда и конфигурация успешно импортированы.")
        st.session_state.ui_module_loaded_print_once = True

except ImportError as e:
    print(f"UI: Не удалось импортировать основные бэкенд-модули: {e}. Приложение не может запуститься.")
    sys.exit(1) 

def load_tesa_label_map_for_ui(config_ref: Any) -> Dict[int, str]:
    """
    Загружает словарь меток TESA из JSON-файла,
    путь к которому указан в конфигурационном модуле.

    Args:
        config_ref (Any): Ссылка на импортированный модуль config.

    Returns:
        Dict[int, str]: Словарь с картой меток TESA.
    """
    if not hasattr(config_ref, 'TESA_LABEL_MAP_PATH'):
        print("UI : TESA_LABEL_MAP_PATH не найден в config_module_ref, используется заглушка для меток.")
        return {0: "POS", 1: "NEG", 2: "NEU"}
    tesa_label_map_path = config_ref.TESA_LABEL_MAP_PATH
    default_map = {0: "POS", 1: "NEG", 2: "NEU"}
    try:
        with open(tesa_label_map_path, 'r', encoding='utf-8') as f:
            tesa_maps_json = json.load(f)
            id2label = {int(k): v for k, v in tesa_maps_json.get('id2label', {}).items()}
            if not id2label:
                print(f"UI: Словарь 'id2label' не найден или пуст в файле {tesa_label_map_path}. Используется заглушка.")
                return default_map
            return id2label
    except FileNotFoundError:
        print(f"UI: Файл словаря меток TESA не найден по пути: {tesa_label_map_path}. Используется заглушка.")
        return default_map
    except Exception as e:
        print(f"UI: Ошибка при загрузке/чтении файла словаря меток TESA {tesa_label_map_path}: {e}. Используется заглушка.")
        return default_map

def initialize_app_state_and_services():
    """
    Инициализирует состояние сессии Streamlit необходимыми 
    значениями по умолчанию и создает экземпляр `AnalysisService`.
    """
    if 'app_initialized' not in st.session_state:
        defaults = {
            'tesa_id2label': load_tesa_label_map_for_ui(config_module),
            'current_group_name': "Не определена",
            'last_group_input': getattr(config_module, 'DEFAULT_GROUP_IDENTIFIERS', [""])[0],
            'last_processed_group_input': "", 
            'last_processed_group_name': "Не определена",
            'last_processed_start_date': None, 
            'last_processed_end_date': None, 
            'ui_start_date': max(datetime.date.today() - datetime.timedelta(days=6), datetime.date.today() - datetime.timedelta(days=365*5)),
            'ui_end_date': datetime.date.today(),
            'initial_summary': None,
            'initial_detailed_mentions': None,
            'current_summary_display': None,
            'current_detailed_mentions_display': None,
            'analysis_triggered_and_pending': False,
            'multiselect_key_counter': 0,
            'selected_entities_for_merge': [], 
            'person_selected_for_details_dropdown': None,
            'analysis_service_instance': None,
            'app_initialization_error': None}
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value        
        try:
            st.session_state.analysis_service_instance = AnalysisService()
            print("UI INFO: AnalysisService успешно инициализирован.")
        except Exception as e_init_service:
            error_msg = f"UI КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать AnalysisService: {e_init_service}. Функционал анализа будет недоступен."
            print(error_msg)
            st.session_state.app_initialization_error = error_msg
        st.session_state.app_initialized = True

initialize_app_state_and_services()

# Вспомогательная функция (можно разместить в начале app_ui.py)
def clean_vk_identifier_for_api(input_str: str) -> str:
    """
    Очищает введенную строку для получения ID или короткого имени (screen_name),
    пригодного для передачи в методы VK API, ожидающие group_id или user_id.
    """
    if not input_str:
        return ""
    cleaned_id = input_str.strip()
    if cleaned_id.startswith("http://"):
        cleaned_id = cleaned_id[7:]
    elif cleaned_id.startswith("https://"):
        cleaned_id = cleaned_id[8:]
    if cleaned_id.startswith("vk.com/"):
        cleaned_id = cleaned_id[7:]
    elif cleaned_id.startswith("m.vk.com/"): 
        cleaned_id = cleaned_id[9:]
    if "/" in cleaned_id:
        cleaned_id = cleaned_id.split("/")[0]
    if "?" in cleaned_id:
        cleaned_id = cleaned_id.split("?")[0]
    if "#" in cleaned_id:
        cleaned_id = cleaned_id.split("#")[0]
    _identifier = input_str.strip()
    if _identifier.startswith("https://"): _identifier = _identifier[8:]
    elif _identifier.startswith("http://"): _identifier = _identifier[7:]
    if _identifier.startswith("vk.com/"): _identifier = _identifier[7:]
    elif _identifier.startswith("m.vk.com/"): _identifier = _identifier[9:]
    if "/" in _identifier: _identifier = _identifier.split("/")[0]
    if "?" in _identifier: _identifier = _identifier.split("?")[0]
    if "#" in _identifier: _identifier = _identifier.split("#")[0]
    if not _identifier:
        print(f"UI WARN: Не удалось извлечь идентификатор из '{input_str}'")
        return ""
    return _identifier

async def fetch_display_group_name_ui_wrapper(group_id_or_url_ui: str):
    """
    Асинхронно получает и отображает имя группы ВКонтакте по ее ID или URL.

    Использует AsyncVKAPI для запроса к VK API. Результат (имя группы
    или сообщение об ошибке) сохраняется в st.session_state.current_group_name.

    Args:
        group_id_or_url_ui (str): Строка, содержащая ID, короткое имя или URL группы VK.
    """
    if not group_id_or_url_ui:
        st.session_state.current_group_name = "ID группы не указан"
        return
    if not hasattr(config_module, 'VK_SERVICE_TOKEN') or not config_module.VK_SERVICE_TOKEN or \
       not hasattr(config_module, 'VK_API_BASE_URL') or not config_module.VK_API_BASE_URL or \
       not hasattr(config_module, 'VK_API_VERSION'):
        st.session_state.current_group_name = "Ошибка: Конфигурация VK API неполная в config.py."
        print("UI: VK_SERVICE_TOKEN, VK_API_BASE_URL или VK_API_VERSION не найдены в config_module.")
        return
    
    identifier = clean_vk_identifier_for_api(group_id_or_url_ui)
    
    try:
        async with aiohttp.ClientSession() as session: 
            vk_api_client_ui = AsyncVKAPI(session=session, token=config_module.VK_SERVICE_TOKEN,
                                          api_version=config_module.VK_API_VERSION, api_base_url=config_module.VK_API_BASE_URL)
            
            group_data_list = await vk_api_client_ui.groups_getById(group_id=identifier, fields="name,screen_name")
            
            if group_data_list and isinstance(group_data_list, list) and group_data_list:
                group_info = group_data_list[0]
                st.session_state.current_group_name = group_info.get("name", f"Группа '{identifier}'")
                print(f"UI INFO: Имя группы '{st.session_state.current_group_name}' для ID/домена '{identifier}' получено.")
            else:
                st.session_state.current_group_name = f"Группа '{identifier}' не найдена или API вернул пустой ответ."
                print(f"UI WARNING: Группа '{identifier}' не найдена или API вернул пустой результат для groups.getById.")
    except Exception as e:
        st.session_state.current_group_name = f"Ошибка ({type(e).__name__}) при проверке имени '{identifier}'"
        print(f"UI ERROR при проверке имени группы '{identifier}': {e}")        

def load_detailed_results_from_file(filepath: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Загружает детализированные результаты анализа 
    из указанного JSONL файла.

    Args:
        filepath (str): Путь к JSONL файлу с детализированными результатами.

    Returns:
        Tuple[List[Dict[str, Any]], List[str]]:
            - detailed_mentions: Список словарей, где каждый словарь - одно упоминание.
            - processed_group_names: Список уникальных имен групп, найденных в данных.
            Возвращает пустые списки в случае ошибки или если файл не найден.
    """
    detailed_mentions = []
    processed_group_names = set()
    if not filepath or not os.path.exists(filepath):
        return detailed_mentions, list(processed_group_names)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f):
                try:
                    mention_data = json.loads(line)
                    detailed_mentions.append(mention_data)
                    if mention_data.get("group_name"):
                        processed_group_names.add(mention_data["group_name"])
                except json.JSONDecodeError:
                    print(f"UI: Ошибка декодирования JSON в файле детальных результатов, строка {line_number+1}: {line.strip()}")
    except Exception as e:
        print(f"UI: Ошибка чтения файла детальных результатов {filepath}: {e}")
    return detailed_mentions, list(processed_group_names)

# --- Функции для рендеринга компонентов пользовательского интерфейса ---
def render_sidebar():
    """
    Отрисовывает боковую панель с элементами управления
    для ввода параметров анализа.
    Также содержит кнопку для запуска анализа.
    """
    with st.sidebar:
        st.markdown("# **Параметры анализа**") 
        default_group_id_ui = (hasattr(config_module, 'DEFAULT_GROUP_IDENTIFIERS') and 
                               len(config_module.DEFAULT_GROUP_IDENTIFIERS) > 0 and 
                               config_module.DEFAULT_GROUP_IDENTIFIERS[0]) or "zlo43"        
        st.markdown("## **Целевая группа ВКонтакте**") 
        group_url_or_id_input_val = st.text_input(
            "Ссылка или короткое имя группы:", 
            value=st.session_state.last_group_input or default_group_id_ui,
            help="Например, 'zlo43' или 'lentach'",
            key="group_input_widget_sidebar_key_v5" ) 
        if group_url_or_id_input_val != st.session_state.last_group_input:
            st.session_state.last_group_input = group_url_or_id_input_val        
        if st.button("Проверить имя группы", key="check_group_name_button_sidebar_key_v5"):
            if st.session_state.last_group_input:
                asyncio.run(fetch_display_group_name_ui_wrapper(st.session_state.last_group_input))
                st.rerun()
        if st.session_state.current_group_name and st.session_state.current_group_name != "Не определена":
            st.caption(f"Выбранная группа: **{st.session_state.current_group_name}**")
        elif st.session_state.last_group_input : 
            st.caption(f"Для группы '{st.session_state.last_group_input}' имя не проверено.")
        else:
            st.caption("Введите ID группы и нажмите 'Проверить имя группы'.")
        st.markdown("---")
        st.markdown("## **Период анализа**")
        today = datetime.date.today()        
        five_years_ago = today - datetime.timedelta(days=365 * 5)         
        start_date_selected = st.date_input( 
            "Начальная дата:", value=st.session_state.ui_start_date,
            min_value=five_years_ago, max_value=today, key="start_date_widget_sidebar_key_v5")
        if start_date_selected != st.session_state.ui_start_date:
            st.session_state.ui_start_date = start_date_selected
            if st.session_state.ui_end_date < st.session_state.ui_start_date:
                st.session_state.ui_end_date = st.session_state.ui_start_date
            if (st.session_state.ui_end_date - st.session_state.ui_start_date).days > 365: 
                st.session_state.ui_end_date = st.session_state.ui_start_date + datetime.timedelta(days=365)
                if st.session_state.ui_end_date > today: st.session_state.ui_end_date = today
            st.rerun()
        actual_max_end_date_for_widget = min(today, st.session_state.ui_start_date + datetime.timedelta(days=365))
        end_date_selected = st.date_input( 
            "Конечная дата:", value=st.session_state.ui_end_date,
            min_value=st.session_state.ui_start_date, max_value=actual_max_end_date_for_widget, key="end_date_widget_sidebar_key_v5")
        if end_date_selected != st.session_state.ui_end_date:
            st.session_state.ui_end_date = end_date_selected
            st.rerun()        
        period_is_invalid = (st.session_state.ui_end_date - st.session_state.ui_start_date).days > 365 or \
                            st.session_state.ui_start_date > st.session_state.ui_end_date
        if (st.session_state.ui_end_date - st.session_state.ui_start_date).days > 365:
            st.warning("Период не должен превышать 1 год.")
        if st.session_state.ui_start_date > st.session_state.ui_end_date:
            st.error("Начальная дата не может быть позже конечной.")        
        analysis_possible = st.session_state.analysis_service_instance is not None and \
                            st.session_state.last_group_input and \
                            st.session_state.current_group_name != "Не определена" and \
                            "Не удалось" not in st.session_state.current_group_name and \
                            "Ошибка" not in st.session_state.current_group_name
        if st.button("Начать анализ", type="primary", disabled=period_is_invalid or not analysis_possible, key="start_analysis_button_sidebar_key_v5"):
            if not analysis_possible:
                st.warning("Пожалуйста, введите корректный ID группы, проверьте имя и убедитесь, что нет ошибок инициализации.")
            else:
                st.session_state.current_summary_display = None
                st.session_state.current_detailed_mentions_display = None
                st.session_state.initial_summary = None
                st.session_state.analysis_triggered_and_pending = True
                st.session_state.selected_entities_for_merge = []
                st.session_state.multiselect_key_counter += 1
                st.session_state.person_selected_for_details_dropdown = None
                raw_group_input_from_state = st.session_state.last_group_input 
                cleaned_group_id_for_processing = clean_vk_identifier_for_api(raw_group_input_from_state)
                st.session_state.last_processed_group_input = cleaned_group_id_for_processing
                st.session_state.last_processed_group_name = st.session_state.current_group_name
                st.session_state.last_processed_start_date = st.session_state.ui_start_date
                st.session_state.last_processed_end_date = st.session_state.ui_end_date
                st.rerun() 
                
def render_report_header():
    """
    Отрисовывает заголовок отчета в основной области UI,
    если анализ был проведен и есть данные для отображения.
    Включает имя группы, период анализа и общее количество найденных персон.
    """
    if st.session_state.last_processed_group_name != "Не определена" and st.session_state.current_summary_display is not None:
        st.header(f"Отчет по группе: {st.session_state.last_processed_group_name}")
        if st.session_state.last_processed_start_date and st.session_state.last_processed_end_date:
            st.subheader(f"Период: {st.session_state.last_processed_start_date.strftime('%d.%m.%Y')} - {st.session_state.last_processed_end_date.strftime('%d.%m.%Y')}")
        total_unique_persons = len(st.session_state.current_summary_display.keys())
        st.success(f"### **Найдено мнений о {total_unique_persons} уникальных персонах**") 
        st.markdown("---")

def render_top10_summary(summary_data: Dict[str, Any], tesa_labels: Dict[int, str]):
    """
    Отрисовывает секцию с топ упоминаемыми персонами.

    Args:
        summary_data (Dict[str, Any]): Агрегированные данные анализа 
                                       (ключ 'summary_by_entity_date' из ответа сервиса).
        tesa_labels (Dict[int, str]): Словарь для сопоставления ID меток тональности с их именами.
    """
    if not summary_data: st.write("Нет данных для отображения топа."); return
    st.subheader("Топ-10 упоминаемых персон и их тональность")
    entity_total_counts = Counter()
    person_sentiment_overall = {}
    pos_label = tesa_labels.get(0, "POS"); neg_label = tesa_labels.get(1, "NEG"); neu_label = tesa_labels.get(2, "NEU")
    for person, date_data in summary_data.items():
        pos, neg, neu = 0,0,0
        for _, counts in date_data.items(): pos += counts.get(pos_label,0); neg += counts.get(neg_label,0); neu += counts.get(neu_label,0)
        entity_total_counts[person] = pos + neg + neu
        person_sentiment_overall[person] = {pos_label:pos, neg_label:neg, neu_label:neu}    
    top_n = 10 
    if not entity_total_counts: st.write("Данные для топа персон отсутствуют."); return
    most_common_persons = entity_total_counts.most_common(top_n)
    df_top_persons_data = [{"Персона": p, "Всего упоминаний": tc,
                            "Позитивных": person_sentiment_overall.get(p,{}).get(pos_label,0),
                            "Негативных": person_sentiment_overall.get(p,{}).get(neg_label,0),
                            "Нейтральных": person_sentiment_overall.get(p,{}).get(neu_label,0)} 
                           for p, tc in most_common_persons]
    df_top = pd.DataFrame(df_top_persons_data).set_index("Персона")
    if not df_top.empty:
        st.dataframe(df_top, use_container_width=True)
        st.markdown("#### Общее количество упоминаний (Топ-10)")
        st.bar_chart(df_top[["Всего упоминаний"]], height=300)
        st.markdown("#### Распределение тональностей (Топ-10)")
        st.bar_chart(df_top[["Позитивных", "Негативных", "Нейтральных"]], height=400)
_multiselect_key_for_callback = ""

def update_multiselect_selection_callback():
    """
    Колбэк-функция для виджета st.multiselect.
    Обновляет st.session_state.selected_entities_for_merge текущим выбором
    пользователя из виджета st.multiselect.
    """
    if _multiselect_key_for_callback in st.session_state:
        st.session_state.selected_entities_for_merge = st.session_state[_multiselect_key_for_callback]
        
def update_selectbox_details_callback(selectbox_key: str):
    """
    Колбэк-функция для виджета st.selectbox (выбор персоны для деталей).
    Обновляет st.session_state.person_selected_for_details_dropdown
    текущим выбором пользователя.

    Args:
        selectbox_key (str): Ключ виджета `st.selectbox`, значение которого нужно прочитать.
    """
    if selectbox_key in st.session_state:
        selected_value = st.session_state[selectbox_key]
        st.session_state.person_selected_for_details_dropdown = selected_value if selected_value else None 

def render_main_report_table_and_merge(summary_data: Dict[str, Any], 
                                       detailed_mentions_data: List[Dict[str, Any]], 
                                       tesa_labels: Dict[int, str], 
                                       service_instance: Union[AnalysisService, None]):
    """
    Отрисовывает сводную таблицу со всеми найденными персонами
    и предоставляет элементы управления для их поиска и объединения.

    Args:
        summary_data (Dict[str, Any]): Агрегированные данные анализа.
        detailed_mentions_data (List[Dict[str, Any]]): Список детализированных упоминаний.
        tesa_labels (Dict[int, str]): Словарь меток тональности.
        service_instance (Union[AnalysisService, None]): Экземпляр сервиса анализа
                                                         для выполнения операции объединения.
                                                         Если None, функция объединения будет недоступна.
    """
    global _multiselect_key_for_callback 
    if not summary_data: st.write("Нет данных для основного отчета."); return    
    st.markdown("---"); st.subheader("🔎 Сводная таблица и объединение персон")
    search_query = st.text_input("Поиск по персонам в таблице:", key="entity_search_main_table_key_v5")
    summary_table_data_for_df = []
    pos_label = tesa_labels.get(0, "POS"); neg_label = tesa_labels.get(1, "NEG"); neu_label = tesa_labels.get(2, "NEU")
    for person, date_data_map in summary_data.items():
        total_mentions, pos, neg, neu = 0,0,0,0
        for counts in date_data_map.values():
            p_c = counts.get(pos_label,0); n_c = counts.get(neg_label,0); u_c = counts.get(neu_label,0)
            pos+=p_c; neg+=n_c; neu+=u_c; total_mentions+=(p_c+n_c+u_c)
        summary_table_data_for_df.append({"Персона":person, "Всего упоминаний":total_mentions, "Позитивных":pos, "Негативных":neg, "Нейтральных":neu})    
    if not summary_table_data_for_df: st.write("Нет данных для таблицы."); return
    df_full = pd.DataFrame(summary_table_data_for_df).sort_values(by="Всего упоминаний", ascending=False).reset_index(drop=True)
    df_display = df_full[df_full["Персона"].str.contains(search_query, case=False, na=False)] if search_query else df_full
    st.dataframe(df_display.set_index("Персона"), height=400, use_container_width=True)
    st.markdown("---"); st.markdown("### Объединение персон")
    options = df_display["Персона"].tolist()    
    _multiselect_key_for_callback = f"entities_multiselect_key_v5_{st.session_state.multiselect_key_counter}" 
    valid_current_selection = [s for s in st.session_state.selected_entities_for_merge if s in options]     
    st.multiselect("Выберите персоны для объединения (>1):", 
                   options=options, default=valid_current_selection, 
                   key=_multiselect_key_for_callback, on_change=update_multiselect_selection_callback)
    if st.session_state.initial_summary is not None:
        if st.button("Сбросить все объединения", key="reset_btn_v5_global"):
            st.session_state.current_summary_display = copy.deepcopy(st.session_state.initial_summary)
            st.session_state.current_detailed_mentions_display = copy.deepcopy(st.session_state.initial_detailed_mentions or [])
            st.session_state.selected_entities_for_merge = [] 
            st.session_state.multiselect_key_counter +=1     
            if st.session_state.person_selected_for_details_dropdown and \
               st.session_state.person_selected_for_details_dropdown not in st.session_state.current_summary_display:
                st.session_state.person_selected_for_details_dropdown = None
            st.info("Объединения сброшены."); st.rerun()

    if len(st.session_state.selected_entities_for_merge) > 1:
        canon_name = st.text_input("Каноническое имя:", 
                                   value=max(st.session_state.selected_entities_for_merge, key=len, default=""), 
                                   key="canon_name_input_v5") 
        if st.button("Объединить выбранные", key="merge_btn_v5", type="primary"):
            if canon_name.strip() and service_instance:
                new_sum, new_det = service_instance.reaggregate_with_aliases(
                    st.session_state.current_summary_display,
                    st.session_state.current_detailed_mentions_display or [],
                    st.session_state.selected_entities_for_merge, canon_name.strip())
                st.session_state.current_summary_display = new_sum
                st.session_state.current_detailed_mentions_display = new_det
                st.session_state.selected_entities_for_merge = [] 
                st.session_state.multiselect_key_counter +=1 
                if st.session_state.person_selected_for_details_dropdown and \
                   (st.session_state.person_selected_for_details_dropdown in st.session_state.selected_entities_for_merge or \
                    st.session_state.person_selected_for_details_dropdown not in st.session_state.current_summary_display):
                     st.session_state.person_selected_for_details_dropdown = None                
                st.rerun()
            elif not service_instance:
                 st.error("Сервис анализа не доступен для объединения.")
            else: st.warning("Укажите каноническое имя.")

def render_person_details_expander(summary_data: Dict[str, Any], 
                                   detailed_mentions_data: List[Dict[str, Any]], 
                                   tesa_labels: Dict[int, str]):
    """
    Отрисовывает секцию для детального просмотра информации по выбранной персоне.

    Args:
        summary_data (Dict[str, Any]): Агрегированные данные анализа.
        detailed_mentions_data (List[Dict[str, Any]]): Список детализированных упоминаний.
        tesa_labels (Dict[int, str]): Словарь меток тональности.
    """
    if not summary_data or len(summary_data) == 0: return 
    st.markdown("---"); st.subheader("📜 Детальный анализ персоны")        
    person_options = [""] + sorted(list(summary_data.keys())) 
    selectbox_key = f"person_details_selectbox_v5_{st.session_state.multiselect_key_counter}" 
    current_person_for_details = st.session_state.person_selected_for_details_dropdown
    if current_person_for_details not in person_options:
        current_person_for_details = "" 
        if st.session_state.person_selected_for_details_dropdown is not None: 
            st.session_state.person_selected_for_details_dropdown = None 
    st.selectbox("Выберите персону для просмотра деталей:", 
                 options=person_options, 
                 index=person_options.index(current_person_for_details) if current_person_for_details in person_options else 0,
                 key=selectbox_key,
                 on_change=update_selectbox_details_callback,
                 args=(selectbox_key,))     
    if st.session_state.person_selected_for_details_dropdown:
        person_to_show = st.session_state.person_selected_for_details_dropdown
        with st.expander(f"Детали по персоне: {person_to_show}", expanded=True):
            st.markdown(f"### Динамика упоминаний для: {person_to_show}")
            person_s_data = summary_data.get(person_to_show,{})
            if person_s_data:
                dates, pos_c, neg_c, neu_c = [],[],[],[]
                pos_label = tesa_labels.get(0, "POS"); neg_label = tesa_labels.get(1, "NEG"); neu_label = tesa_labels.get(2, "NEU")
                for date_str, counts in sorted(person_s_data.items()): 
                    try:
                        dates.append(pd.to_datetime(date_str).date()) 
                        pos_c.append(counts.get(pos_label,0)); neg_c.append(counts.get(neg_label,0)); neu_c.append(counts.get(neu_label,0))
                    except Exception as e_date_conv: 
                        print(f"UI WARNING: Ошибка преобразования даты '{date_str}' для графика персоны '{person_to_show}': {e_date_conv}")
                        pass 
                if dates:
                    df_dyn = pd.DataFrame({'Дата':dates, pos_label:pos_c, neg_label:neg_c, neu_label:neu_c}).set_index('Дата')
                    st.line_chart(df_dyn, use_container_width=True)
                else: st.write("Нет данных для построения графика динамики.")
            else: st.write("Нет данных о динамике для этой персоны.")            
            st.markdown(f"### Тексты упоминаний для: {person_to_show}")
            p_mentions = [m for m in (detailed_mentions_data or []) if m.get("entity_normalized") == person_to_show]
            if p_mentions:
                max_texts_to_show_val = max(1, min(10, len(p_mentions))) 
                max_texts_slider_max = max(25, len(p_mentions))                
                max_texts = st.slider("Количество отображаемых текстов:", 
                                      min_value=1, 
                                      max_value=max_texts_slider_max, 
                                      value=max_texts_to_show_val, 
                                      key=f"txt_slider_{person_to_show.replace(' ','_')}_v5")                 
                for i,m in enumerate(p_mentions[:max_texts]):
                    st.markdown(f"**Упоминание {i+1}**")
                    c1,c2,c3 = st.columns(3) 
                    with c1: st.caption(f"Тип: {m.get('source_type','N/A')}")
                    with c2: 
                        ts = m.get('timestamp')
                        date_str_display = "N/A"
                        if ts:
                            try: date_str_display = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
                            except: date_str_display = f"{ts} (ошибка формата)"
                        st.caption(f"Дата: {date_str_display}")
                    with c3: st.caption(f"Тональность: {m.get('polarity','N/A')}")
                    st.markdown(f"> _{m.get('text_preview','Нет текста')}_") 
                    st.markdown("---") 
            else: st.write("Нет текстов упоминаний для этой персоны.")
            
# --- Точка входа и управление потоком UI ---
st.set_page_config(page_title="Анализ тональности персон в ВК", layout="wide", initial_sidebar_state="expanded")
st.title("🗣️ Анализ тональности к персонам в группах ВКонтакте")
st.markdown("Приложение для объектно-ориентированного анализа тональности текстов из групп ВКонтакте.")
st.markdown("---")
if st.session_state.app_initialization_error:
    st.error(f"Критическая ошибка инициализации приложения: {st.session_state.app_initialization_error}")
    st.info("Пожалуйста, проверьте конфигурацию моделей, пути к файлам и сообщения в консоли.")
    st.stop() 
render_sidebar()
if st.session_state.analysis_triggered_and_pending:
    st.session_state.analysis_triggered_and_pending = False 
    group_identifier_to_process = st.session_state.last_processed_group_input
    service = st.session_state.analysis_service_instance 
    
    if st.session_state.last_processed_group_name != "Не определена":
        st.header(f"Обработка группы: {st.session_state.last_processed_group_name}")
        if st.session_state.last_processed_start_date and st.session_state.last_processed_end_date:
            st.subheader(f"Период: {st.session_state.last_processed_start_date.strftime('%d.%m.%Y')} - {st.session_state.last_processed_end_date.strftime('%d.%m.%Y')}")
        st.markdown("---")
    
    if not service:
        st.error("Сервис анализа не инициализирован. Запуск невозможен.")
    else:
        with st.spinner(f"Идет анализ группы '{st.session_state.last_processed_group_name}' (ID/домен: {group_identifier_to_process})... Пожалуйста, подождите."):
            try:
                analysis_results = asyncio.run(service.run_full_analysis(
                    group_identifiers_str=group_identifier_to_process,
                    date_start_str=st.session_state.last_processed_start_date.strftime("%Y-%m-%d"),
                    date_end_str=st.session_state.last_processed_end_date.strftime("%Y-%m-%d")
                ))
                if "error" in analysis_results: st.error(f"Ошибка анализа: {analysis_results['error']}")
                elif "message" in analysis_results and not analysis_results.get("summary"):
                    st.info(analysis_results['message'])
                    st.session_state.current_summary_display = {}; st.session_state.current_detailed_mentions_display = []
                    st.session_state.initial_summary = {}; st.session_state.initial_detailed_mentions = []
                elif analysis_results.get("summary") is not None:
                    summary_from_backend = analysis_results["summary"]
                    detailed_file = analysis_results.get("detailed_results_file")
                    mentions_from_file, _ = load_detailed_results_from_file(detailed_file) if detailed_file else ([], [])
                    st.session_state.initial_summary = copy.deepcopy(summary_from_backend)
                    st.session_state.initial_detailed_mentions = copy.deepcopy(mentions_from_file)
                    st.session_state.current_summary_display = copy.deepcopy(summary_from_backend)
                    st.session_state.current_detailed_mentions_display = copy.deepcopy(mentions_from_file)
                else: 
                    st.info("Анализ завершен. Не найдено данных для отображения.")
                    st.session_state.current_summary_display = {}; st.session_state.current_detailed_mentions_display = []
                    st.session_state.initial_summary = {}; st.session_state.initial_detailed_mentions = []
                
                st.rerun() 

            except Exception as e_runtime: 
                st.error(f"Произошла непредвиденная ошибка во время выполнения анализа: {e_runtime}")
                import traceback; st.text(traceback.format_exc())
                st.session_state.current_summary_display = None 
                st.session_state.initial_summary = None
                st.rerun() 
                
render_report_header()
            
if st.session_state.current_summary_display is not None:
    service_to_pass = st.session_state.analysis_service_instance 
    render_top10_summary(st.session_state.current_summary_display, st.session_state.tesa_id2label)    
    render_main_report_table_and_merge( 
        st.session_state.current_summary_display,
        st.session_state.current_detailed_mentions_display,
        st.session_state.tesa_id2label,
        service_to_pass )
    render_person_details_expander(
        st.session_state.current_summary_display,
        st.session_state.current_detailed_mentions_display,
        st.session_state.tesa_id2label)
elif not st.session_state.analysis_triggered_and_pending and st.session_state.app_initialization_error is None:
    st.info("Задайте параметры анализа в боковой панели и нажмите 'Начать анализ', чтобы увидеть результаты.")
elif st.session_state.app_initialization_error:
    st.error(f"Критическая ошибка инициализации приложения: {st.session_state.app_initialization_error}")