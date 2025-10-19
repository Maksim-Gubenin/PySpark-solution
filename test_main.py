from typing import Any, List, Optional, Tuple

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from pyspark.testing import assertDataFrameEqual

from main import ProductCategoryService


class TestProductCategoryService:
    """
    Тесты для ProductCategoryService.

    Проверяют корректность работы сервиса по обработке связей между продуктами и категориями
    """

    @pytest.fixture(scope="class")
    def spark_session(self) -> SparkSession:
        """
        Создает Spark сессию для выполнения тестов.
        """

        return (
            SparkSession.builder.master("local[1]")
            .appName("TestProductCategory")
            .getOrCreate()
        )

    @pytest.fixture
    def service(self, spark_session: SparkSession) -> ProductCategoryService:
        """
        Создает экземпляр тестируемого сервиса.
        """

        return ProductCategoryService(spark_session)

    @pytest.fixture
    def sample_dataframes(
        self, spark_session: SparkSession
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Создает тестовые датафреймы для использования в тестах.
        """

        products_data: List[Tuple[int, str]] = [
            (1, "Laptop"),
            (2, "Smartphone"),
            (3, "Book"),
            (4, "Uncategorized Product"),
        ]
        products_df: DataFrame = spark_session.createDataFrame(
            products_data, ["product_id", "product_name"]
        )

        categories_data: List[Tuple[int, str]] = [
            (1, "Electronics"),
            (2, "Computers"),
            (3, "Literature"),
        ]
        categories_df: DataFrame = spark_session.createDataFrame(
            categories_data, ["category_id", "category_name"]
        )

        links_data: List[Tuple[int, int]] = [
            (1, 1),
            (1, 2),
            (2, 1),
            (3, 3),
        ]
        links_df: DataFrame = spark_session.createDataFrame(
            links_data, ["product_id", "category_id"]
        )

        return products_df, categories_df, links_df

    def test_validation_with_correct_dataframes(
        self,
        service: ProductCategoryService,
        sample_dataframes: Tuple[DataFrame, DataFrame, DataFrame],
    ) -> None:
        """
        Проверяет успешную валидацию корректных датафреймов.
        """

        products_df, categories_df, links_df = sample_dataframes
        service.validate_dataframes(products_df, categories_df, links_df)

    def test_validation_with_missing_product_columns(
        self,
        service: ProductCategoryService,
        sample_dataframes: Tuple[DataFrame, DataFrame, DataFrame],
    ) -> None:
        """
        Проверяет валидацию при отсутствии колонок в датафрейме продуктов.
        """

        products_df, categories_df, links_df = sample_dataframes
        invalid_products_df: DataFrame = products_df.drop("product_name")
        with pytest.raises(ValueError):
            service.validate_dataframes(invalid_products_df, categories_df, links_df)

    def test_validation_with_missing_category_columns(
        self,
        service: ProductCategoryService,
        sample_dataframes: Tuple[DataFrame, DataFrame, DataFrame],
    ) -> None:
        """
        Проверяет валидацию при отсутствии колонок в датафрейме категорий.
        """

        products_df, categories_df, links_df = sample_dataframes
        invalid_categories_df: DataFrame = categories_df.drop("category_name")
        with pytest.raises(ValueError):
            service.validate_dataframes(products_df, invalid_categories_df, links_df)

    def test_validation_with_missing_link_columns(
        self,
        service: ProductCategoryService,
        sample_dataframes: Tuple[DataFrame, DataFrame, DataFrame],
    ) -> None:
        """
        Проверяет валидацию при отсутствии колонок в датафрейме связей.
        """

        products_df, categories_df, links_df = sample_dataframes
        invalid_links_df: DataFrame = links_df.drop("category_id")
        with pytest.raises(ValueError):
            service.validate_dataframes(products_df, categories_df, invalid_links_df)

    def test_get_products_with_categories_returns_correct_pairs(
        self,
        service: ProductCategoryService,
        sample_dataframes: Tuple[DataFrame, DataFrame, DataFrame],
        spark_session: SparkSession,
    ) -> None:
        """
        Проверяет корректность возвращаемых пар продукт-категория.
        """

        products_df, categories_df, links_df = sample_dataframes
        result_df: DataFrame = service.get_products_with_categories(
            products_df, categories_df, links_df
        )

        expected_data: List[Tuple[str, Optional[str]]] = [
            ("Laptop", "Electronics"),
            ("Laptop", "Computers"),
            ("Smartphone", "Electronics"),
            ("Book", "Literature"),
            ("Uncategorized Product", None),
        ]
        expected_df: DataFrame = spark_session.createDataFrame(
            expected_data, ["product_name", "category_name"]
        )

        result_sorted: DataFrame = result_df.orderBy("product_name", "category_name")
        expected_sorted: DataFrame = expected_df.orderBy(
            "product_name", "category_name"
        )
        assertDataFrameEqual(result_sorted, expected_sorted)

    def test_products_without_categories_are_included(
        self,
        service: ProductCategoryService,
        sample_dataframes: Tuple[DataFrame, DataFrame, DataFrame],
    ) -> None:
        """
        Проверяет включение продуктов без категорий в результат.
        """

        products_df, categories_df, links_df = sample_dataframes
        result_df: DataFrame = service.get_products_with_categories(
            products_df, categories_df, links_df
        )

        products_without_categories: DataFrame = result_df.filter(
            "category_name IS NULL"
        )
        assert products_without_categories.count() == 1

        uncategorized_product: str = products_without_categories.collect()[0][
            "product_name"
        ]
        assert uncategorized_product == "Uncategorized Product"

    def test_products_with_multiple_categories_are_duplicated(
        self,
        service: ProductCategoryService,
        sample_dataframes: Tuple[DataFrame, DataFrame, DataFrame],
    ) -> None:
        """
        Проверяет дублирование продуктов с несколькими категориями.
        """

        products_df, categories_df, links_df = sample_dataframes
        result_df: DataFrame = service.get_products_with_categories(
            products_df, categories_df, links_df
        )

        laptop_rows: List[Any] = result_df.filter("product_name = 'Laptop'").collect()
        assert len(laptop_rows) == 2

        laptop_categories: List[str] = [row["category_name"] for row in laptop_rows]
        assert set(laptop_categories) == {"Electronics", "Computers"}

    def test_empty_links_dataframe_returns_all_products_without_categories(
        self,
        service: ProductCategoryService,
        sample_dataframes: Tuple[DataFrame, DataFrame, DataFrame],
        spark_session: SparkSession,
    ) -> None:
        """
        Проверяет поведение при пустом датафрейме связей.
        """

        products_df, categories_df, _ = sample_dataframes

        empty_links_df: DataFrame = spark_session.createDataFrame(
            [],
            StructType(
                [
                    StructField("product_id", IntegerType(), True),
                    StructField("category_id", IntegerType(), True),
                ]
            ),
        )

        result_df: DataFrame = service.get_products_with_categories(
            products_df, categories_df, empty_links_df
        )
        assert result_df.filter("category_name IS NOT NULL").count() == 0
        assert result_df.count() == products_df.count()

    def test_empty_products_dataframe_returns_empty_result(
        self,
        service: ProductCategoryService,
        sample_dataframes: Tuple[DataFrame, DataFrame, DataFrame],
        spark_session: SparkSession,
    ) -> None:
        """
        Проверяет поведение при пустом датафрейме продуктов.
        """

        _, categories_df, links_df = sample_dataframes

        empty_products_df: DataFrame = spark_session.createDataFrame(
            [],
            StructType(
                [
                    StructField("product_id", IntegerType(), True),
                    StructField("product_name", StringType(), True),
                ]
            ),
        )

        result_df: DataFrame = service.get_products_with_categories(
            empty_products_df, categories_df, links_df
        )
        assert result_df.count() == 0

    def test_main_execution(
        self, service: ProductCategoryService, spark_session: SparkSession
    ) -> None:
        """
        Тестирует выполнение main блока для увеличения coverage.
        """

        products_df: DataFrame
        categories_df: DataFrame
        links_df: DataFrame
        products_df, categories_df, links_df = service.create_example_dataframes()
        result_df: DataFrame = service.get_products_with_categories(
            products_df, categories_df, links_df
        )
        assert result_df.count() > 0
