# backend/vk_parser/parser.py

import asyncio
import aiohttp
import json
import datetime
import sys 
import os
from typing import List, Dict, Tuple, Any, Union

try:
    import config 
except ImportError as e:
    print(f"PARSER CRITICAL: Не удалось импортировать 'config'. Ошибка: {e}. Завершение работы.")
    sys.exit(1)

class VKAPIError(Exception):
    """
    Исключение, для ошибок при взаисодействии с VK API.
    """
    def __init__(self, message: str, error_code: Union[int, None] = None):
        """
        Args:
            message (str): Описание ошибки.
            error_code (Union[int, None], optional): Код ошибки от VK API.
        """
        self.error_code = error_code
        self.message = message
        super().__init__(f"VK API Error (Code: {self.error_code}): {self.message}" if error_code else f"VK API Error: {self.message}")

class AsyncVKAPI:
    """
    Асинхронный клиент для взаимодействия с VK API.
    """
    def __init__(self, 
                 session: aiohttp.ClientSession, 
                 token: str, 
                 api_version: str,
                 api_base_url: str, 
                 max_retries: int = 3,
                 base_retry_delay_s: float = 1.0,
                 rate_limit_delay_s: float = 10.0 ):
        """
        Инициализирует асинхронный клиент VK API.

        Args:
            session (aiohttp.ClientSession): Активная сессия aiohttp.
            token (str): Сервисный ключ доступа VK.
            api_version (str): Версия используемого VK API.
            api_base_url (str): Базовый URL для запросов к VK API.
            max_retries (int, optional): Максимальное количество повторных попыток при ошибках.
            base_retry_delay_s (float, optional): Начальная задержка в секундах для повтора
                                                  при ошибке "слишком много запросов".
            rate_limit_delay_s (float, optional): Задержка в секундах при ошибке "rate limit".
        """
        self.session = session
        self.token = token
        self.api_version = api_version
        self.api_base_url = api_base_url
        self.max_retries = max_retries
        self.base_retry_delay_s = base_retry_delay_s
        self.rate_limit_delay_s = rate_limit_delay_s
        print(f"AsyncVKAPI инициализирован для v{api_version}.")

    async def _call_method(self, method_name: str, params: dict = None) -> Union[Dict[str, Any], List[Any], None]:
        """
        Выполнение вызова метода VK API.
        
        Args:
            method_name (str): Имя вызываемого метода VK API.
            params (dict, optional): Словарь параметров для метода API.

        Returns:
            Union[Dict[str, Any], List[Any], None]: Ответ от VK API.

        Raises:
            VKAPIError: В случае ошибок VK API  или если не удалось получить ответ после всех повторных попыток.
        """
        if params is None: params = {}        
        params["access_token"] = self.token
        params["v"] = self.api_version
        url = f"{self.api_base_url}{method_name}"         
        for attempt in range(self.max_retries):
            try:            
                async with self.session.post(url, data=params) as response:
                    if 'application/json' not in response.headers.get('Content-Type', ''):
                        text_response = await response.text()
                        print(f"PARSER WARNING: Неожиданный Content-Type от VK API для {method_name}: {response.headers.get('Content-Type')}. Ответ: {text_response[:200]}")
                        if attempt < self.max_retries - 1:
                             await asyncio.sleep(self.base_retry_delay_s * (2**attempt))
                             continue
                        raise VKAPIError(f"Unexpected Content-Type: {response.headers.get('Content-Type')}")
                    data = await response.json()
                    response.raise_for_status() 
                    if "error" in data:
                        error_info = data["error"]
                        error_code = error_info.get("error_code")
                        error_msg = error_info.get("error_msg", "Unknown API error")
                        print(f"PARSER WARNING: VK API Error ({method_name}): Code {error_code}, Msg: '{error_msg}'. Attempt {attempt+1}/{self.max_retries}.")
                        if error_code == 6: 
                            sleep_time = self.base_retry_delay_s * (2 ** attempt); await asyncio.sleep(sleep_time)
                        elif error_code == 29: 
                            sleep_time = self.rate_limit_delay_s + (attempt * 5); await asyncio.sleep(sleep_time)
                        else: raise VKAPIError(error_msg, error_code=error_code)
                        if attempt == self.max_retries - 1: raise VKAPIError(f"VK API Error ({method_name}) не устранена: {error_msg}", error_code=error_code)
                        continue                    
                    if hasattr(config, 'DELAY_AFTER_API_CALL_SECONDS'):
                        await asyncio.sleep(config.DELAY_AFTER_API_CALL_SECONDS)
                    return data.get("response")            
            except aiohttp.ClientError as e: 
                print(f"PARSER WARNING: HTTP/Network Error ({method_name}): {e}. Attempt {attempt+1}/{self.max_retries}.")
                if attempt == self.max_retries - 1:
                    print(f"PARSER CRITICAL: HTTP/Network Error ({method_name}) не устранена: {e}")
                    raise VKAPIError(f"Network/HTTP error after retries: {e}") from e 
                await asyncio.sleep(self.base_retry_delay_s * (2 ** attempt))        
        final_msg = f"Failed to call {method_name} after {self.max_retries} retries."
        print(f"PARSER ERROR: {final_msg}")
        raise VKAPIError(final_msg)
    async def groups_getById(self, group_id: str, fields: str = None):
        """ Вызывает метод VK API groups.getById. """
        params = {"group_id": group_id}
        if fields: params["fields"] = fields
        return await self._call_method("groups.getById", params)
    async def wall_get(self, owner_id: int, count: int, offset: int, **kwargs):
        """ Вызывает метод VK API wall.get с фильтром 'owner'. """
        params = {"owner_id": owner_id, "count": count, "offset": offset, "filter": "owner", **kwargs}
        return await self._call_method("wall.get", params)
    async def wall_getComments(self, owner_id: int, post_id: int, count: int, offset: int, **kwargs):
        """ Вызывает метод VK API wall.getComments. """
        params = {"owner_id": owner_id, "post_id": post_id, "count": count, "offset": offset, **kwargs}
        params.setdefault('sort', 'asc')
        params.setdefault('thread_items_count', 0) 
        return await self._call_method("wall.getComments", params)
    
class VKGroupProcessor:
    """
    Класс, для парсинга  одной указанной VK группы за определенный период времени.
    """
    def __init__(self, vk_api_client: AsyncVKAPI, group_identifier: str,
                 start_timestamp: int, end_timestamp: int, semaphore: asyncio.Semaphore,
                 posts_chunk_size: int, comments_chunk_size: int, 
                 max_comments_per_post: Union[int, None]):
        """
        Инициализирует процессор для парсинга одной группы.

        Args:
            vk_api_client (AsyncVKAPI): Экземпляр клиента для взаимодействия с VK API.
            group_identifier (str): ID или короткое имя группы VK.
            start_timestamp (int): Начальная метка времени для парсинга.
            end_timestamp (int): Конечная метка времени для парсинга.
            semaphore (asyncio.Semaphore): Семафор для ограничения одновременных запросов к API в рамках обработки этой группы.
            posts_chunk_size (int): Количество постов, запрашиваемых за один вызов API.
            comments_chunk_size (int): Количество комментариев, запрашиваемых за один вызов API.
            max_comments_per_post (Union[int, None]): Максимальное количество комментариев для парсинга с одного поста.
        """
        self.api = vk_api_client
        self.group_identifier = group_identifier
        self.start_ts = start_timestamp
        self.end_ts = end_timestamp
        self.semaphore = semaphore
        self.posts_chunk_size = posts_chunk_size
        self.comments_chunk_size = comments_chunk_size
        self.max_comments_per_post = max_comments_per_post
        self.group_info: Union[Dict[str, Any], None] = None
        self.vk_group_id_numeric: Union[int, None] = None
        
    async def _fetch_group_info(self) -> bool:
        """
        Получение и сохранения информации о группе.

        Returns:
            bool: True, если информация о группе успешно получена, иначе False.
        """
        print(f"PARSER INFO: Получение информации о группе: {self.group_identifier}")
        try:
            group_data_list = await self.api.groups_getById(group_id=str(self.group_identifier), fields="name,screen_name")
            if group_data_list and isinstance(group_data_list, list) and group_data_list:
                group = group_data_list[0]
                self.group_info = {'vk_group_id': group['id'], 'screen_name': group.get('screen_name'), 'name': group.get('name')}
                self.vk_group_id_numeric = group['id']
                print(f"PARSER INFO: Информация для группы '{self.group_info.get('name', self.vk_group_id_numeric)}' (ID: {self.vk_group_id_numeric}) получена.")
                return True
        except VKAPIError as e: print(f"PARSER ERROR: VKAPIError при получении информации о группе '{self.group_identifier}': {e}")
        except Exception as e: print(f"PARSER ERROR: Неожиданная ошибка при получении информации о группе '{self.group_identifier}': {e}")
        return False
        
    async def _parse_posts(self) -> List[Dict[str, Any]]:
        """
        Парсинг постов группы за указанный период.

        Returns:
            List[Dict[str, Any]]: Список словарей, где каждый словарь представляет пост.
        """
        if self.vk_group_id_numeric is None: return []
        collected_posts_data = []
        offset = 0
        print(f"PARSER INFO: Парсинг постов для группы ID {self.vk_group_id_numeric}...")
        while True:
            async with self.semaphore:
                try:
                    wall_chunk = await self.api.wall_get(owner_id=-self.vk_group_id_numeric, count=self.posts_chunk_size, offset=offset)
                    if not wall_chunk or not wall_chunk.get('items'): print(f"PARSER INFO: Больше постов не найдено для гр. {self.vk_group_id_numeric} (offset {offset})."); break
                    items = wall_chunk['items']
                    if not items: break
                    oldest_post_date_in_chunk = items[-1]['date']
                    stop_fetching_posts = oldest_post_date_in_chunk < self.start_ts
                    for post_item in items:
                        if post_item['date'] > self.end_ts: continue
                        if post_item['date'] >= self.start_ts:
                            post_data = {'vk_post_id': post_item['id'], 'owner_id': post_item['owner_id'],
                                         'date': post_item['date'], 'text': post_item.get('text', '').strip(),
                                         'comments_api_count': post_item.get('comments', {}).get('count', 0)}
                            if post_data['text']: collected_posts_data.append(post_data)
                    if stop_fetching_posts: print(f"PARSER INFO: Достигнут конец периода для гр. {self.vk_group_id_numeric}."); break
                    offset += self.posts_chunk_size
                    if len(items) < self.posts_chunk_size: print(f"PARSER INFO: Достигнут конец стены гр. {self.vk_group_id_numeric}."); break
                except VKAPIError as e: print(f"PARSER ERROR: VKAPIError при парсинге постов гр. {self.vk_group_id_numeric} (offset {offset}): {e}"); break
                except Exception as e: print(f"PARSER ERROR: Неожиданная ошибка при парсинге постов гр. {self.vk_group_id_numeric} (offset {offset}): {e}"); break
        print(f"PARSER INFO: Сбор постов для гр. {self.vk_group_id_numeric} завершен. Собрано: {len(collected_posts_data)}.")
        return collected_posts_data

    async def _parse_comments_for_post(self, vk_post_id: int) -> List[Dict[str, Any]]:
        """
        Парсинга комментариев к одному посту за указанный период.

        Args:
            vk_post_id (int): ID поста, для которого собираются комментарии.

        Returns:
            List[Dict[str, Any]]: Список словарей, где каждый словарь представляет комментарий.
        """
        if self.vk_group_id_numeric is None: return []
        collected_comments_data = []
        offset, fetched_count = 0, 0
        while True:
            if self.max_comments_per_post is not None and fetched_count >= self.max_comments_per_post: break
            async with self.semaphore:
                try:
                    count_to_request = self.comments_chunk_size
                    if self.max_comments_per_post is not None:
                        remaining = self.max_comments_per_post - fetched_count
                        if remaining <= 0: break
                        count_to_request = min(self.comments_chunk_size, remaining)
                    comments_chunk = await self.api.wall_getComments(
                        owner_id=-self.vk_group_id_numeric, post_id=vk_post_id, count=count_to_request, offset=offset) 
                    if not comments_chunk or not comments_chunk.get('items'): break
                    items = comments_chunk['items']
                    if not items: break
                    stop_for_this_post = False
                    for item in items:
                        if item['date'] > self.end_ts: stop_for_this_post = True; break
                        if item['date'] < self.start_ts: continue
                        text = item.get('text', '').strip()
                        if text:
                            collected_comments_data.append({'vk_comment_id': item['id'], 'from_id': item.get('from_id'),
                                                            'date': item['date'], 'text': text})
                            fetched_count += 1
                            if self.max_comments_per_post is not None and fetched_count >= self.max_comments_per_post:
                                stop_for_this_post = True; break
                    if stop_for_this_post: break
                    offset += len(items)
                    if len(items) < count_to_request: break
                except VKAPIError as e:
                    if e.error_code == 18: print(f"PARSER WARNING: Пост {vk_post_id} удален/скрыт или комментарии закрыты.")
                    else: print(f"PARSER ERROR: VKAPIError при парсинге комментариев к посту {vk_post_id}: {e}")
                    break
                except Exception as e: print(f"PARSER ERROR: Неожиданная ошибка при парсинге комментариев поста {vk_post_id}: {e}"); break
        return collected_comments_data

    async def process_and_write_to_file(self, writer_lock: asyncio.Lock, output_file_path: str) -> int:
        """
        Полный цикл обработки для одной группы: получение информации о группе,
        парсинг постов, асинхронный парсинг комментариев для этих постов и
        запись всех собранных данных в указанный файл.

        Args:
            writer_lock (asyncio.Lock): Асинхронный лок для синхронизации записи в файл.
            output_file_path (str): Путь к выходному файлу JSONL.

        Returns:
            int: Количество успешно обработанных и сохраненных постов для данной группы.
        """
        print(f"PARSER INFO: --- Начало полной обработки группы: {self.group_identifier} ---")
        if not await self._fetch_group_info() or self.vk_group_id_numeric is None: return 0
        posts_in_period = await self._parse_posts()
        if not posts_in_period: print(f"PARSER INFO: Постов не найдено для группы {self.group_identifier}."); return 0
        all_group_data_to_write = []
        for post_data in posts_in_period:
            comments_for_this_post = []
            if post_data['comments_api_count'] > 0:
                try: comments_for_this_post = await self._parse_comments_for_post(post_data['vk_post_id'])
                except Exception as e_comm: print(f"PARSER ERROR: Ошибка при получении комментариев для поста {post_data['vk_post_id']}: {e_comm}")
            all_group_data_to_write.append({
                "vk_group_id": self.vk_group_id_numeric,
                "group_screen_name": self.group_info.get('screen_name') if self.group_info else None,
                "group_name": self.group_info.get('name') if self.group_info else None,
                **post_data, "comments": comments_for_this_post   })
        group_posts_saved = 0
        if all_group_data_to_write:
            async with writer_lock:
                try:
                    with open(output_file_path, 'a', encoding='utf-8') as f_out:
                        for entry_to_write in reversed(all_group_data_to_write): # reversed для хронологического порядка
                            f_out.write(json.dumps(entry_to_write, ensure_ascii=False) + '\n')
                            group_posts_saved += 1
                    print(f"PARSER INFO: Сохранено {group_posts_saved} постов для группы {self.vk_group_id_numeric}.")
                except IOError as e_io:
                    print(f"PARSER CRITICAL: Ошибка записи в файл {output_file_path} для группы {self.group_identifier}: {e_io}")
                    return 0
        print(f"PARSER INFO: --- Завершение обработки группы: {self.group_identifier}. Сохранено: {group_posts_saved} ---")
        return group_posts_saved

async def fetch_vk_data(group_identifiers: List[str], 
                        start_date_obj: datetime.date, 
                        end_date_obj: datetime.date) -> str:
    """
    Основная асинхронная функция для сбора данных из VK.

    Args:
        group_identifiers (List[str]): Список ID или коротких имен групп VK.
        start_date_obj (datetime.date): Начальная дата периода для парсинга.
        end_date_obj (datetime.date): Конечная дата периода для парсинга.

    Returns:
        str: Путь к файлу JSONL, содержащему собранные данные.

    Raises:
        ValueError: Если начальная дата позже конечной, или если VK_SERVICE_TOKEN не установлен.
        IOError: Если не удалось создать/очистить выходной файл.
        VKAPIError: Если происходят неустранимые ошибки при взаимодействии с VK API.
    """
    start_ts = int(datetime.datetime.combine(start_date_obj, datetime.time.min).timestamp())
    end_ts = int(datetime.datetime.combine(end_date_obj, datetime.time.max).timestamp())
    if start_ts > end_ts:
        print("PARSER ERROR: Начальная дата не может быть позже конечной.")
        raise ValueError("Начальная дата не может быть позже конечной.")
    output_file = config.PARSED_DATA_OUTPUT_FILE
    try:
        with open(output_file, 'w', encoding='utf-8') as f_clear: print(f"PARSER INFO: Файл результатов {output_file} очищен/создан.")
    except IOError as e: print(f"PARSER CRITICAL: Не удалось очистить/создать файл {output_file}: {e}"); raise 
    total_posts_saved_overall = 0
    session_timeout = aiohttp.ClientTimeout(total=300) 
    async with aiohttp.ClientSession(timeout=session_timeout) as http_session:
        if not config.VK_SERVICE_TOKEN: 
            print("PARSER CRITICAL: VK_SERVICE_TOKEN не установлен в конфигурации! Парсинг невозможен.")
            raise ValueError("VK_SERVICE_TOKEN не установлен.")
        vk_api_client = AsyncVKAPI(
            http_session, config.VK_SERVICE_TOKEN, config.VK_API_VERSION, config.VK_API_BASE_URL )
        writer_lock = asyncio.Lock()
        group_processing_tasks = []
        for group_identifier in group_identifiers:
            group_semaphore = asyncio.Semaphore(config.CONCURRENT_API_REQUESTS_PER_GROUP_SEMAPHORE)
            processor = VKGroupProcessor(
                vk_api_client=vk_api_client, group_identifier=group_identifier,
                start_timestamp=start_ts, end_timestamp=end_ts, semaphore=group_semaphore,
                posts_chunk_size=config.POSTS_CHUNK_SIZE, comments_chunk_size=config.COMMENTS_CHUNK_SIZE,
                max_comments_per_post=config.MAX_COMMENTS_PER_POST_SESSION)
            task = processor.process_and_write_to_file(writer_lock, output_file)
            group_processing_tasks.append(task)
            if len(group_identifiers) > 1: await asyncio.sleep(0.05)
        results = await asyncio.gather(*group_processing_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception): print(f"PARSER ERROR: Ошибка при обработке группы '{group_identifiers[i]}': {result}")
            elif isinstance(result, int): total_posts_saved_overall += result
    print(f"\nPARSER INFO: Парсинг завершен. Всего сохранено {total_posts_saved_overall} постов в файл {output_file}.")
    return output_file