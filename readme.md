## Сгенерить сайт-документацию yandex-cloud 
```shell
yfm -i . -o ./output
```
## Настроить python окружение
```shell
python3 -m venv .venv
source .venv/bin/activate 

# requirements
pip install beautifulsoup4
pip install html2text
pip install pymilvus
pip install 'pymilvus[model]'

streamlit run ui.py
```

## Поднять milvus и залить схему
