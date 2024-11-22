# Використовуємо офіційний Python-образ
FROM python:3.10-slim

# Копіюємо проект
WORKDIR /app
COPY . /app

# Встановлюємо залежності
RUN pip install --no-cache-dir -r requirements.txt

# Експонуємо порт
EXPOSE 5000

# Запускаємо додаток
CMD ["python", "app.py"]
