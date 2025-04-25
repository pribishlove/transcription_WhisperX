

--- Скачиваем на свой ПК:
1. Python3.10: https://www.python.org/downloads/release/python-31011/
2. NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-downloads


--- Создаём виртуальное окружение:
```sh
python -m venv .venv
```


--- Входим в виртуальное окружение:
А) Через переключение версий python справа внизу VSCode с "3.10 Global" на "3.10 (.venv)"
После этого открыть новый терминал и должно появиться уведомление, что venv скрыто активирован.
Б) Если не получилось сделать А), то второй подход(aka "Каменный век"):
**Windows:**
```sh
source .venv/Scripts/activate
#иногда не работает и надо без source просто написать: .venv/Scripts/activate
```
**Linux/Mac:**
```sh
source .venv/bin/activate
```


--- Whisper и зависимости.
ВАРИАНТ А(чат гпт)):
--- Устанавливаем Pytorch перед whisperx: https://pytorch.org
```sh
# Почему именно эта версия? тип другие выдают ошибки да, но где написанно что эта окажется норм? как нагуглил?
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # Для GPU
```
```sh
# Почему не через pip install whisper
pip install git+https://github.com/m-bain/whisperX.git
```
```sh
pip install numpy torchaudio transformers ffmpeg-python silero-vad
```
```sh
pip install python-dotenv
```
ВАРИАНТ Б(у меня сработало):
- Установка аналогично обычному whisper:
1. Скачиваем whisperX:  pip install git+https://github.com/m-bain/whisperX.git
2. удаляем torch(потому что если сначала скачать торч не той версии, то whisper при обновлении скачет торч на cpu):   python -m pip uninstall torch
3. устанавливаем torch с CUDA:   https://pytorch.org/get-started/locally/


ДАЛЕЕ:
--- Создаём себе Токен Hugging Face для скачивания модели для диаризации спикера:
1. Создаём аккаунт на HuggingFace.
2. Идём в Настройки => Access Tokens(https://huggingface.co/settings/tokens) => Create new token =>
{
Token name = whisperx
Все галочки оставьте в изначальном состоянии
В "Repositories permissions" добавьте: pyannote/speaker-diarization-3.1 и pyannote/segmentation-3.0
}
=> CreateToken => Скопирывать токен и сохранить его куда-нибудь.
3. Получить доступ к моделям. Надо зайти на каждую из ссылок и получить доступ к моделям(для этого заполнить данные вверху страницы).
3.1 https://huggingface.co/pyannote/segmentation-3.0
3.2 https://huggingface.co/pyannote/speaker-diarization-3.1
4. Создаём в проекте в корневом каталоге файл с названием ".env"
Вставляем в него строку:
```python
HUGGINGFACE_TOKEN="hf_..."
```
5. Вставьте в "" свой токен.



--- Установка cuDNN
1. Авторизуйтесь на сайте **NVIDIA**
2. Перейдите в раздел архива и скачайте **cuDNN v8.9.6**:
   🔗 [Скачать cuDNN v8.9.6](https://developer.nvidia.com/rdp/cudnn-archive)
3. Распакуйте архив и скопируйте файлы в соответствующие папки:
| Папка в архиве | Куда скопировать |
|---------------|----------------|
| **bin/** | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin` |
| **lib/** | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64` |
| **include/** | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include` |





---

## Добавление аудиофайла
Скопируйте аудиофайл в рабочую папку и укажите его название в коде **transcriber.py**.

---

## Настройка скрипта:
Выставьте настройки в transcriber.py:
Те, что в подпунктах: "установи значения перед запуском" и "зависит от устройства".