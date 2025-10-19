from typing import Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit


class ProductCategoryService:
    """
    Сервис для обработки связей между продуктами и категориями в PySpark.

    Реализует логику для получения всех пар "продукт-категория" и продуктов без категорий.
    Использует JOIN операции для объединения данных.
    """

    def __init__(self, spark: SparkSession) -> None:
        """
        Инициализирует сервис с Spark сессией и определяет обязательные колонки.

        :param spark: Spark сессия для выполнения операций
        :type spark: SparkSession
        """

        self.spark: SparkSession = spark

        self.required_product_cols: set[str] = {"product_id", "product_name"}
        self.required_category_cols: set[str] = {"category_id", "category_name"}
        self.required_link_cols: set[str] = {"product_id", "category_id"}

    def validate_dataframes(
        self, products_df: DataFrame, categories_df: DataFrame, links_df: DataFrame
    ) -> None:
        """
        Проверяет наличие необходимых колонок во входных датафреймах.

        Проверяет наличие всех обязательных колонок в каждом датафрейме.

        :param products_df: датафрейм с продуктами
        :type products_df: DataFrame
        :param categories_df: датафрейм с категориями
        :type categories_df: DataFrame
        :param links_df: датафрейм со связями продуктов и категорий
        :type links_df: DataFrame

        :raises ValueError: если в каком-либо датафрейме отсутствуют обязательные колонки
        """

        if not self.required_product_cols.issubset(products_df.columns):
            raise ValueError(
                f"Products DataFrame must contain columns: {self.required_product_cols}"
            )

        if not self.required_category_cols.issubset(categories_df.columns):
            raise ValueError(
                f"Categories DataFrame must contain columns: {self.required_category_cols}"
            )

        if not self.required_link_cols.issubset(links_df.columns):
            raise ValueError(
                f"Links DataFrame must contain columns: {self.required_link_cols}"
            )

    def get_products_with_categories(
        self,
        products_df: DataFrame,
        categories_df: DataFrame,
        product_category_links_df: DataFrame,
    ) -> DataFrame:
        """
        Возвращает все пары "Имя продукта – Имя категории" и продукты без категорий.

        Реализует двухэтапный подход:
        1. JOIN операциями находим все существующие пары продукт-категория
        2. LEFT ANTI JOIN находим продукты без категорий
        3. UNION объединяет результаты в один датафрейм

        :param products_df: датафрейм с продуктами (должен содержать product_id, product_name)
        :type products_df: DataFrame
        :param categories_df: датафрейм с категориями (должен содержать category_id, category_name)
        :type categories_df: DataFrame
        :param product_category_links_df: датафрейм со связями (должен содержать product_id, category_id)
        :type product_category_links_df: DataFrame

        :return: датафрейм с колонками product_name и category_name
        :rtype: DataFrame

        :raises ValueError: если входные датафреймы не прошли валидацию
        """
        self.validate_dataframes(
            products_df=products_df,
            categories_df=categories_df,
            links_df=product_category_links_df,
        )

        product_category_pairs: DataFrame = (
            product_category_links_df.join(products_df, "product_id")
            .join(categories_df, "category_id")
            .select("product_name", "category_name")
        )

        products_without_categories: DataFrame = products_df.join(
            product_category_links_df, "product_id", "left_anti"
        ).select(col("product_name"), lit(None).alias("category_name"))

        result_df: DataFrame = product_category_pairs.union(products_without_categories)

        return result_df

    def create_example_dataframes(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Создает примеры датафреймов для демонстрации работы сервиса.

        Создает тестовые данные, включающие:
        - Продукты с категориями (Laptop, Smartphone, Epson, Video card)
        - Продукт без категорий (War and peace)
        - Категории без продуктов (Certificate)
        - Связи многие-ко-многим между продуктами и категориями

        :return: кортеж с датафреймами (products_df, categories_df, links_df)
        :rtype: Tuple[DataFrame, DataFrame, DataFrame]
        """
        products_data: list[tuple[int, str]] = [
            (1, "Laptop"),
            (2, "Smartphone"),
            (3, "Epson"),
            (4, "Video card"),
            (5, "War and peace"),
        ]
        products_df: DataFrame = self.spark.createDataFrame(
            products_data, ["product_id", "product_name"]
        )

        categories_data: list[tuple[int, str]] = [
            (1, "Electronics"),
            (2, "Computers"),
            (3, "Mobile"),
            (4, "Printer"),
            (5, "Certificate"),
        ]
        categories_df: DataFrame = self.spark.createDataFrame(
            categories_data, ["category_id", "category_name"]
        )

        links_data: list[tuple[int, int]] = [
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 4),
            (4, 1),
        ]
        links_df: DataFrame = self.spark.createDataFrame(
            links_data, ["product_id", "category_id"]
        )

        return products_df, categories_df, links_df


if __name__ == "__main__":
    spark = SparkSession.builder.appName("ProductCategoryExample").getOrCreate()

    service = ProductCategoryService(spark)
    products_df, categories_df, links_df = service.create_example_dataframes()

    result_df = service.get_products_with_categories(
        products_df, categories_df, links_df
    )
    result_df.show()

    spark.stop()
