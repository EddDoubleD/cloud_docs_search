# Cloud Docs Search

Поиск по документации Yandex Cloud с гибридным семантическим поиском: dense- и sparse-эмбеддинги (BGE-M3) в [Milvus](https://milvus.io/). Веб-интерфейс на Streamlit позволяет сравнивать результаты **Dense**, **Sparse** и **Hybrid** поиска в одном экране.

## Что делает проект

1. **Парсинг документации** — обходит сгенерированный сайт документации (HTML), извлекает заголовки, URL и текст в Markdown, сохраняет по одному JSON на страницу.
2. **Индексация** — разбивает текст на чанки, считает dense- и sparse-эмбеддинги моделью BGE-M3 и записывает векторы в коллекцию Milvus.
3. **Поиск** — гибридный поиск по запросу с подсветкой совпадений и ссылками на страницы на yandex.cloud.

## Архитектура

```
Yandex Cloud docs (исходники)
        ↓
   yfm (сборка сайта)
        ↓
   output/ru (HTML)
        ↓
   html_parser.py  →  output_ru_parsed/ (JSON)
        ↓
   index.py  →  Milvus (ru_docs)
        ↓
   ui.py (Streamlit)  →  Dense / Sparse / Hybrid поиск
```

## Требования

- Python 3.10+
- Docker и Docker Compose (для Milvus)
- Для эмбеддингов: Mac с MPS или GPU (в коде используется `device="mps"`; на Linux без GPU можно заменить на `device="cpu"`)

## Установка и запуск

### 1. Виртуальное окружение и зависимости

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install beautifulsoup4 html2text pymilvus 'pymilvus[model]' langchain-text-splitters streamlit
```

### 2. Сборка сайта документации Yandex Cloud

Из репозитория с исходниками документации (где есть конфиг для YFM):

```bash
yfm -i . -o ./output
```

В результате появится папка `output` (в т.ч. `output/ru` с HTML).

### 3. Запуск Milvus и создание коллекции

Из корня проекта:

```bash
cd docker
docker compose up -d
```

Дождитесь готовности сервисов (порты: Milvus — 19530, MinIO — 9000/9001). Затем создайте коллекцию и индексы:

```bash
cd ..
python db/schema.py
```

При необходимости проверьте, что коллекция создана:

```bash
python db/schema_check.py
```

### 4. Парсинг HTML в JSON

Укажите путь к папке с HTML (по умолчанию — `./output/ru`) и каталог для JSON:

```bash
python html_parser.py [путь/к/output/ru] -o ./output_ru_parsed -j 3
```

- `-o` — директория для JSON-файлов (по умолчанию `output_ru_parsed`)
- `-j` — число потоков (по умолчанию 3)

### 5. Индексация в Milvus

```bash
python index.py ./output_ru_parsed
```

Можно указать другую папку с JSON. Скрипт обходит все `.json`, разбивает текст на чанки, считает эмбеддинги BGE-M3 и вставляет записи в коллекцию `ru_docs`.

### 6. Веб-интерфейс поиска

```bash
streamlit run ui.py
```

В браузере откроется страница с полем запроса и тремя колонками: **Dense**, **Sparse**, **Hybrid**. По запросу выводятся фрагменты с подсветкой совпадений и ссылками на документацию на yandex.cloud.

## Структура проекта

| Путь | Назначение |
|------|------------|
| `html_parser.py` | Парсинг HTML из `output/ru`, извлечение title/canonical/text (Markdown) → JSON |
| `index.py` | Обход JSON, чанкинг, BGE-M3, вставка в Milvus (`ru_docs`) |
| `ui.py` | Streamlit-приложение: гибридный поиск и подсветка совпадений |
| `db/schema.py` | Создание коллекции Milvus и индексов (dense + sparse) |
| `db/schema_check.py` | Проверка списка коллекций |
| `docker/docker-compose.yml` | Milvus standalone (etcd, MinIO, Milvus) |

## Настройки

- **Milvus:** по умолчанию `http://localhost:19530`, токен `root:Milvus` (задаётся в `index.py`, `ui.py`, `db/schema.py`).
- **Устройство для BGE-M3:** в коде используется `device="mps"` (Apple Silicon). На машине без MPS замените на `device="cpu"` в `index.py`, `ui.py` и `db/schema.py`.
- **Чанки:** в `index.py` — `RecursiveCharacterTextSplitter` с `chunk_size=512`, `chunk_overlap=50`. При необходимости измените под свой объём и стиль текста.

## Лицензия

Используйте в соответствии с лицензией исходной документации Yandex Cloud и зависимостей (Milvus, BGE-M3, Streamlit и др.).
