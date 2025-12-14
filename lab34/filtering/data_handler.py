import pandas as pd
import random
from pathlib import Path
from typing import Dict, List, Optional

from .config import USERS_DATA_PATH, FILMS_DATA_PATH

class DataProcessor:
    """
    Обработчик датасета MovieLens для загрузки и обработки данных о фильмах и оценках.
    """
    ratings_df: pd.DataFrame
    user_item_table: pd.DataFrame
    movie_titles: pd.DataFrame
    movie_genres: Dict[int, List[str]]
    genre_names: List[str]


    def __init__(self) -> None:
        self.ratings_df = pd.DataFrame(
            columns=["user_id", "movie_id", "rating", "timestamp"]
        )
        self.user_item_table = pd.DataFrame()
        self.movie_titles = pd.DataFrame(columns=["movie_id", "title"])
        self.movie_genres = {}
        self.genre_names = [
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ]


    async def load_data(self) -> None:
        """
        Асинхронная загрузка данных из файлов u.data и u.item.
        """
        users_path = Path(USERS_DATA_PATH)
        films_path = Path(FILMS_DATA_PATH)

        try:
            self.ratings_df = pd.read_csv(
                users_path,
                sep="\t",
                names=["user_id", "movie_id", "rating", "timestamp"],
                dtype={"user_id": int, "movie_id": int, "rating": float, "timestamp": int},
                engine="python",
            )
            print(f"Загружено оценок: {len(self.ratings_df)}")
        except Exception as e:
            print(f"Ошибка при загрузке {users_path}: {e}")
            self.ratings_df = pd.DataFrame(
                columns=["user_id", "movie_id", "rating", "timestamp"]
            )
            self.user_item_table = pd.DataFrame()
            return

        if films_path.exists():
            try:
                cols = [
                    "movie_id",
                    "title",
                    "release_date",
                    "video_release_date",
                    "imdb_url",
                ] + self.genre_names
                df_items = pd.read_csv(
                    films_path,
                    sep="|",
                    encoding="latin-1",
                    header=None,
                    names=cols,
                    usecols=list(range(5 + len(self.genre_names))),
                )
                self.movie_titles = df_items[["movie_id", "title"]].copy()

                genres_map: Dict[int, List[str]] = {}
                for _, row in df_items.iterrows():
                    mid = int(row["movie_id"])
                    genres: List[str] = []
                    for g in self.genre_names:
                        try:
                            if int(row[g]) == 1:
                                genres.append(g)
                        except Exception:
                            continue
                    genres_map[mid] = genres
                self.movie_genres = genres_map

                print(f"Загружено названий фильмов: {len(self.movie_titles)}")
            except Exception as e:
                print(f"Ошибка загрузки {films_path}: {e}")
                self.movie_titles = pd.DataFrame(columns=["movie_id", "title"])
                self.movie_genres = {}
        else:
            print(f"Файл {films_path} не найден. Названия фильмов не загружены.")

        await self._create_user_item_table()


    async def _create_user_item_table(self) -> None:
        """
        Создание таблицы [пользователь X фильм] с заполнением нулями.
        """
        if self.ratings_df is None or self.ratings_df.empty:
            self.user_item_table = pd.DataFrame()
            return
        self.user_item_table = self.ratings_df.pivot_table(
            index="user_id",
            columns="movie_id",
            values="rating",
            aggfunc="mean",
        ).fillna(0.0)


    def get_user_ratings(self, user_id: int) -> Dict[int, float]:
        """
        Получение словаря ненулевых рейтингов для заданного пользователя.
        :param user_id: ID пользователя
        :return: Словарь {movie_id: rating} или пустой словарь если пользователь не найден
        """
        if (
            self.user_item_table is None
            or self.user_item_table.empty
            or user_id not in self.user_item_table.index
        ):
            return {}
        user_ratings = self.user_item_table.loc[user_id]
        return {
            int(movie_id): float(rating)
            for movie_id, rating in user_ratings.items()
            if rating > 0
        }


    def get_all_users(self) -> List[int]:
        """
        Получение списка всех ID пользователей.
        :return: Список ID пользователей
        """
        if self.user_item_table is None or self.user_item_table.empty:
            return []
        return list(self.user_item_table.index.tolist())


    def get_movie_title(self, movie_id: int) -> str:
        """
        Получение названия фильма по ID.
        :param movie_id: ID фильма
        :return: Название фильма или 'Фильм {id}' если не найден
        """
        if self.movie_titles is not None and not self.movie_titles.empty:
            row = self.movie_titles[self.movie_titles["movie_id"] == movie_id]
            if not row.empty:
                return str(row["title"].iloc[0])
        return f"Фильм {movie_id}"


    def get_movie_genres(self, movie_id: int) -> List[str]:
        """
        Получение списка жанров фильма.
        :param movie_id: ID фильма
        :return: Список жанров фильма
        """
        return self.movie_genres.get(movie_id, [])


    def get_top_popular_movies(self, n: int = 50) -> List[int]:
        """
        Получение топ-N популярных фильмов по количеству оценок.
        :param n: Количество фильмов для возврата, defaults to 50
        :return: Список ID популярных фильмов
        """
        if self.ratings_df is None or self.ratings_df.empty:
            return []
        counts = self.ratings_df.groupby("movie_id").size()
        return counts.nlargest(n).index.tolist()


    def get_random_movies(self, n: int = 5) -> List[int]:
        """
        Получение N случайных фильмов из датасета.
        :param n: Количество случайных фильмов, defaults to 5
        :return: Список ID случайных фильмов
        """
        if self.movie_titles is not None and not self.movie_titles.empty:
            all_movies = self.movie_titles["movie_id"].tolist()
        elif self.ratings_df is not None and not self.ratings_df.empty:
            all_movies = self.ratings_df["movie_id"].unique().tolist()
        else:
            return []
        return random.sample(all_movies, min(n, len(all_movies)))
