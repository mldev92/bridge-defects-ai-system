import streamlit as st
import shutil
import os
from defects_process import initiation, start_analize, compile_all_info

# Путь к папкам и файлам
output_folder = './output_data'
archive_name = './result.zip'
video_name = './input.mp4'


# Удаление предыдущих данных
def clear_previous_data():
    if os.path.exists(archive_name):
        os.remove(archive_name)

def clear_previous_data2():
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    if os.path.exists(video_name):
        os.remove(video_name)


# Интерфейс Streamlit для загрузки и обработки видео
st.title("Обработка видео с дефектами")

# Загрузка видео
uploaded_video = st.file_uploader("Загрузите видео в формате MP4", type=['mp4'])

if uploaded_video:
    # Сохранение загруженного видео
    with open(video_name, 'wb') as f:
        f.write(uploaded_video.read())

    st.write("Видео загружено успешно.")

    # Запуск обработки
    if st.button('Запустить обработку'):
        # Очистка предыдущих данных
        clear_previous_data()

        # Инициация, анализ и компиляция
        st.write("Обработка видео... ( ~ 350мс/фрэйм или ~ продолжительность видео * 22 )")
        initiation()  # Инициализация моделей и захват видео
        start_analize()  # Анализ видео
        compile_all_info()  # Компиляция результатов в архив

        # Проверка наличия архива
        if os.path.exists(archive_name):
            st.success("Обработка завершена. Вы можете скачать результат.")
            with open(archive_name, 'rb') as f:
                st.download_button("Скачать архив с результатами", f, file_name="results_archive.zip")
        else:
            st.error("Ошибка: Архив результатов не найден.")

        # Удаление временных данных
        clear_previous_data2()
