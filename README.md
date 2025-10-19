# PySpark Product-Category Solution

Решение задачи по обработке связей между продуктами и категориями с использованием PySpark.

## Описание
Проект содержит реализацию сервиса для работы со связями "многие-ко-многим" между продуктами и категориями. Основная задача - получить все пары "продукт-категория" и продукты без категорий в одном датафрейме.

## Структура проекта
```text
pyspark-solution/
├── main.py # Основная логика ProductCategoryService
├── test_main.py # Unit-тесты для проверки функциональности
├── pyproject.toml # Конфигурация проекта и зависимости
└── README.md # Документация
```

## Функциональность
- ProductCategoryService
Основной класс, предоставляющий методы для работы с продуктами и категориями:

- get_products_with_categories() - возвращает все пары "продукт-категория" и продукты без категорий

- validate_dataframes() - проверяет корректность входных датафреймов

- create_example_dataframes() - создает тестовые данные для демонстрации

## Пример данных

### Продукты:

- Laptop (с категориями: Electronics, Computers)

- Smartphone (с категориями: Electronics, Mobile)

- Epson (с категорией: Printer)

- Video card (с категорией: Electronics)

- War and peace (без категорий)

### Категории:

- Electronics

- Computers

- Mobile

- Printer

- Certificate (без продуктов)

## 🚀Использование
Запуск примера
```python
from main import ProductCategoryService
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("ProductCategoryExample").getOrCreate()
service = ProductCategoryService(spark)

# Создание тестовых данных
products_df, categories_df, links_df = service.create_example_dataframes()

# Получение результата
result_df = service.get_products_with_categories(products_df, categories_df, links_df)
result_df.show()
```

### 📚Ожидаемый вывод
```text
+---------------+------------+
|   product_name|category_name|
+---------------+------------+
|         Laptop| Electronics|
|         Laptop|   Computers|
|      Smartphone| Electronics|
|      Smartphone|      Mobile|
|          Epson|     Printer|
|     Video card| Electronics|
|   War and peace|        null|
+---------------+------------+
```
## Тестирование
Проект включает comprehensive unit-тесты, проверяющие:

-Корректность валидации входных данных

-Наличие продуктов без категорий

-Дублирование продуктов с несколькими категориями

-Обработку пустых датафреймов связей

-Корректность возвращаемых пар "продукт-категория"

### Запуск тестов
```bash
  pytest test_main.py -v
```
### Запуск линтеров
```bash
  mypy .
  black . --check
  ruff check
```

## Технические детали

### 📋 Зависимости
- Python >= 3.12

- PySpark >= 4.0.1, < 5.0.0

## ?!Предположения и ограничения
- Все датафреймы должны содержать обязательные колонки: product_id, product_name, category_id, category_name

- Связи хранятся в отдельном датафрейме с колонками product_id, category_id

- Категории без продуктов не включаются в результат (только пары и продукты без категорий)

## 👨‍💻 Автор

Губенин Максим

- Email: maksimgubenin@mail.ru
- GitHub: https://github.com/Maksim-Gubenin