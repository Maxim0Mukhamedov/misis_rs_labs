from typing import Dict, List, Tuple, Optional
import asyncio
import numpy as np
import pandas as pd

from .data_handler import DataProcessor


class CollaborativeFiltering:
    """
    Реализация User-Based Collaborative Filtering с корреляцией Пирсона.
    """

    def __init__(
        self,
        data_processor: DataProcessor,
        min_common_items: int = 3,
        top_k: int = 20,
    ) -> None:
        """
        Инициализация Collaborative Filtering.
        :param data_processor: Обработчик данных
        :param min_common_items: Минимальное количество общих фильмов для расчёта сходства пользователей, defaults = 3
        :param top_k: Количество ближайших соседей для предсказания, defaults = 20
        """
        self.dp = data_processor
        self.min_common_items = min_common_items
        self.top_k = top_k

        # Матрица сходств между пользователями: {user_id: {other_user_id: similarity}}
        self.sim: Dict[int, Dict[int, float]] = {}

        # Средние рейтинги пользователей (для центрирования): {user_id: mean_rating}
        self.user_means: Dict[int, float] = {}

        # Таблица user x item
        self.user_item_table: Optional[pd.DataFrame] = None

        self._built = False
        self._build_lock = asyncio.Lock()

    async def _ensure_built(self) -> None:
        """
        Гарантирует, что матрица сходств построена.
        """
        if self._built:
            return
        async with self._build_lock:
            if not self._built:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.build_user_similarity)
                self._built = True

    def _ensure_user_item_table(self) -> None:
        """
        Гарантирует наличие user_item_table (user x item).
        """
        if self.dp.user_item_table is not None and not self.dp.user_item_table.empty:
            self.user_item_table = self.dp.user_item_table
            return

        ratings_df = self.dp.ratings_df
        self.user_item_table = ratings_df.pivot_table(
            index="user_id", columns="movie_id", values="rating", aggfunc="mean"
        ).fillna(0.0)

    def _pearson_between_rows(
        self,
        row_u: pd.Series,
        row_v: pd.Series,
        min_common: int
    ) -> Optional[float]:
        """
        Корреляция Пирсона между двумя пользователями (ряды user_item_table).
        Учитываются только совместно оцененные фильмы (>0).
        """
        mask = (row_u > 0) & (row_v > 0)
        if int(mask.sum()) < min_common:
            return None

        u = row_u[mask].values.astype(float)
        v = row_v[mask].values.astype(float)

        u_center = u - u.mean()
        v_center = v - v.mean()

        denom = np.sqrt((u_center ** 2).sum()) * np.sqrt((v_center ** 2).sum())
        if denom == 0:
            return None

        sim = float((u_center * v_center).sum() / denom)
        return sim

    def _pearson_virtual_to_user(
        self,
        virtual_user_ratings: Dict[int, float],
        row_v: pd.Series,
        min_common: int
    ) -> Optional[float]:
        """
        Корреляция Пирсона между виртуальным пользователем (dict item->rating)
        и существующим пользователем (ряд таблицы).
        """
        common_items = [
            iid for iid, _ in virtual_user_ratings.items()
            if iid in row_v.index and row_v[iid] > 0
        ]
        if len(common_items) < min_common:
            return None

        u = np.array([virtual_user_ratings[iid] for iid in common_items], dtype=float)
        v = row_v.loc[common_items].values.astype(float)

        u_center = u - u.mean()
        v_center = v - v.mean()

        denom = np.sqrt((u_center ** 2).sum()) * np.sqrt((v_center ** 2).sum())
        if denom == 0:
            return None

        sim = float((u_center * v_center).sum() / denom)
        return sim

    def build_user_similarity(self) -> None:
        """
        Построение матрицы сходств между пользователями.
        """
        print("Начинаю построение матрицы сходства пользователей...")
        self._ensure_user_item_table()
        assert self.user_item_table is not None

        user_ids = self.user_item_table.index.tolist()
        n = len(user_ids)

        sim: Dict[int, Dict[int, float]] = {}

        # Предвычислим средние рейтинги пользователей
        means: Dict[int, float] = {}
        for uid in user_ids:
            row = self.user_item_table.loc[uid]
            rated = row[row > 0]
            means[uid] = float(rated.mean()) if len(rated) > 0 else 3.0
        self.user_means = means

        for i_idx, u_id in enumerate(user_ids):
            sim[u_id] = {}
            if (i_idx + 1) % 100 == 0:
                print(f"  обработано {i_idx + 1}/{n} пользователей...")

            row_u = self.user_item_table.loc[u_id]
            for v_id in user_ids[i_idx + 1:]:
                row_v = self.user_item_table.loc[v_id]
                similarity = self._pearson_between_rows(row_u, row_v, self.min_common_items)
                if similarity is not None and abs(similarity) > 0.1:
                    sim[u_id][v_id] = similarity
                    if v_id not in sim:
                        sim[v_id] = {}
                    sim[v_id][u_id] = similarity

        self.sim = sim
        print("Матрица сходства пользователей построена.")

    def predict_rating(
        self,
        user_ratings: Dict[int, float],
        item_id: int,
        k: Optional[int] = None
    ) -> Optional[float]:
        """
        Предсказание рейтинга виртуального пользователя для заданного фильма.
        :param user_ratings: Оценки виртуального пользователя {movie_id: rating}
        :param item_id: ID целевого фильма
        :param k: Количество соседей для учета, defaults to self.top_k
        :return: Предсказанный рейтинг или None если предсказание невозможно
        """
        if k is None:
            k = self.top_k
        if not user_ratings:
            return None

        self._ensure_user_item_table()
        if self.user_item_table is None or item_id not in self.user_item_table.columns:
            return None

        # Средний рейтинг виртуального пользователя
        target_user_mean = float(np.mean(list(user_ratings.values()))) if len(user_ratings) > 0 else 3.0

        # Пользователи, оценившие данный фильм
        col = self.user_item_table[item_id]
        neighbor_ids = col[col > 0].index.tolist()
        if not neighbor_ids:
            return None

        # Сходство виртуального пользователя с соседями
        sims: List[Tuple[int, float, float]] = []  # (neighbor_id, sim, neighbor_rating_for_item)
        for v_id in neighbor_ids:
            row_v = self.user_item_table.loc[v_id]
            sim_uv = self._pearson_virtual_to_user(user_ratings, row_v, self.min_common_items)
            if sim_uv is not None and sim_uv > 0:
                sims.append((v_id, sim_uv, float(row_v[item_id])))

        if not sims:
            return None

        # Топ-K соседей
        sims.sort(key=lambda x: x[1], reverse=True)
        sims = sims[:k]

        # Взвешенное предсказание: u_mean + sum(sim*(r_vi - v_mean)) / sum(|sim|)
        numerator = 0.0
        denominator = 0.0
        for v_id, sim_uv, r_vi in sims:
            v_mean = self.user_means.get(v_id)
            if v_mean is None:
                row_v = self.user_item_table.loc[v_id]
                rated = row_v[row_v > 0]
                v_mean = float(rated.mean()) if len(rated) > 0 else 3.0
                self.user_means[v_id] = v_mean
            numerator += sim_uv * (r_vi - v_mean)
            denominator += abs(sim_uv)

        if denominator == 0:
            return None

        prediction = target_user_mean + (numerator / denominator)
        prediction = float(max(1.0, min(5.0, prediction)))
        return prediction

    async def generate_recommendations(
        self,
        virtual_user_ratings: Dict[int, float],
        num_recommendations: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Генерация рекомендаций для виртуального пользователя.
        :param virtual_user_ratings: Оценки пользователя
        :param num_recommendations: Количество рекомендаций, defaults to 5
        :return: Список кортежей (movie_id, predicted_rating)
        """
        await self._ensure_built()

        popular = self.dp.get_top_popular_movies(1000)
        watched = set(virtual_user_ratings.keys())
        candidate_ids = [m for m in popular if m not in watched]
        if len(candidate_ids) > 1000:
            candidate_ids = candidate_ids[:1000]

        predicted: List[Tuple[int, float]] = []
        for item_id in candidate_ids:
            pred = self.predict_rating(virtual_user_ratings, item_id)
            if pred is not None and pred > 3.0:
                predicted.append((item_id, pred))

        if not predicted:
            watched = set(virtual_user_ratings.keys())
            popular = self.dp.get_top_popular_movies(num_recommendations * 2)
            result: List[Tuple[int, float]] = []
            for mid in popular:
                if mid not in watched:
                    result.append((mid, 0.0))
                if len(result) >= num_recommendations:
                    break
            return result

        predicted.sort(key=lambda x: x[1], reverse=True)
        return predicted[:num_recommendations]
