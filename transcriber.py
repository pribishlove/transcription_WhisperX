import torch
import whisperx
from dotenv import load_dotenv
import os

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


# Определяем устройство (GPU или CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используем устройство: {device}")

# Загружаем основную модель
model = whisperx.load_model("large", device)

# Путь к аудиофайлу
audio_path = "test.m4a"

# Транскрибация
transcription = model.transcribe(audio_path)
print("\nТекст с таймкодами:", transcription["segments"])

# Загружаем модель для выравнивания
align_model, metadata = whisperx.load_align_model(language_code="ru", device=device)

# Выравниваем текст
aligned_transcription = whisperx.align(
    transcription["segments"], align_model, metadata, audio_path, device
)
print("\nВыравненный текст:", aligned_transcription["segments"])

# Загружаем модель для определения спикеров
diarize_model = whisperx.DiarizationPipeline(use_auth_token=HUGGINGFACE_TOKEN, device=device)

# Выполняем диаризацию (разделение на спикеров)
diarization = diarize_model(audio_path)

# Объединяем с транскрипцией
final_result = whisperx.assign_word_speakers(diarization, aligned_transcription)

# Сохранение результата в файл
with open("transcription_output.txt", "w", encoding="utf-8") as f:
    f.write("Текст с таймкодами:\n")
    for segment in transcription["segments"]:
        f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}\n")

    f.write("\nВыравненный текст:\n")
    for segment in aligned_transcription["segments"]:
        f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}\n")

    f.write("\nРезультат с разделением по спикерам:\n")
    for segment in final_result["segments"]:
        f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] Speaker {segment['speaker']}: {segment['text']}\n")

print("Результаты сохранены в 'transcription_output.txt'")
