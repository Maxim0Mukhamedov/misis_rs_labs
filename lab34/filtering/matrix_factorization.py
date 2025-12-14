from typing import Dict, List, Tuple, Optional
import asyncio
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .data_handler import DataProcessor


@dataclass
class MFConfig:
    n_factors: int = 40            # размерность латентного пространства
    n_epochs: int = 20             # количество эпох SGD
    lr: float = 0.01               # скорость обучения
    reg: float = 0.05              # L2-регуляризация
    min_ratings_to_infer: int = 2  # минимум оценок виртуального пользователя для инференса его вектора
    seed: int = 42                 # для воспроизводимости
    clip_min: float = 1.0          # нижняя граница предсказания
    clip_max: float = 5.0          # верхняя граница предсказания


class MatrixFactorization:
    """
    Рекомендательная система на основе матричной факторизации (MF).
    Обучение по известным пользователям и фильмам (SGD), предсказание для
    виртуального пользователя через решение ridge-задачи на его оценках.
    """

    def __init__(
        self,
        data_processor: DataProcessor,
        config: MFConfig = MFConfig(),
        top_k_candidates: int = 1000
    ) -> None:
        self.dp = data_processor
        self.cfg = config
        self.top_k_candidates = top_k_candidates

        # Маппинги id -> index и обратно
        self.uid2idx: Dict[int, int] = {}
        self.idx2uid: List[int] = []
        self.iid2idx: Dict[int, int] = {}
        self.idx2iid: List[int] = []

        # Параметры MF
        self.P: Optional[np.ndarray] = None  # user_factors (n_users x k)
        self.Q: Optional[np.ndarray] = None  # item_factors (n_items x k)
        self.bu: Optional[np.ndarray] = None # user biases (n_users)
        self.bi: Optional[np.ndarray] = None # item biases (n_items)
        self.mu: float = 0.0                 # глобальное среднее

        # Для быстрого доступа
        self.user_item_table: Optional[pd.DataFrame] = None

        # Флаги
        self._built = False
        self._build_lock = asyncio.Lock()

    def _prepare_data(self) -> None:
        """
        Готовим user-item таблицу, индексацию и numpy-массивы для обучения.
        """
        ratings_df = self.dp.ratings_df

        # Гарантируем наличие user_item_table
        if self.dp.user_item_table is not None and not self.dp.user_item_table.empty:
            self.user_item_table = self.dp.user_item_table
        else:
            self.user_item_table = ratings_df.pivot_table(
                index="user_id", columns="movie_id", values="rating", aggfunc="mean"
            ).fillna(0.0)

        # Идентификаторы
        user_ids = ratings_df["user_id"].unique().tolist()
        item_ids = ratings_df["movie_id"].unique().tolist()
        self.uid2idx = {uid: i for i, uid in enumerate(user_ids)}
        self.idx2uid = user_ids
        self.iid2idx = {iid: i for i, iid in enumerate(item_ids)}
        self.idx2iid = item_ids

        # Глобальное среднее
        self.mu = float(ratings_df["rating"].mean())

    def _init_params(self) -> None:
        """
        Инициализация факторов и биасов.
        """
        rng = np.random.default_rng(self.cfg.seed)
        n_users = len(self.idx2uid)
        n_items = len(self.idx2iid)
        k = self.cfg.n_factors

        # Инициализация малым нормальным шумом
        self.P = rng.normal(0, 0.05, size=(n_users, k)).astype(np.float32)
        self.Q = rng.normal(0, 0.05, size=(n_items, k)).astype(np.float32)
        self.bu = np.zeros(n_users, dtype=np.float32)
        self.bi = np.zeros(n_items, dtype=np.float32)

    def _build_training_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Преобразует DataFrame в numpy-массивы индексов и рейтингов для SGD.
        """
        df = self.dp.ratings_df
        u_idx = df["user_id"].map(self.uid2idx).values.astype(np.int32)
        i_idx = df["movie_id"].map(self.iid2idx).values.astype(np.int32)
        r = df["rating"].values.astype(np.float32)
        return u_idx, i_idx, r

    def train_model(self) -> None:
        """
        Обучение MF с помощью SGD.
        """
        print("Начинаю обучение Matrix Factorization...")
        self._prepare_data()
        self._init_params()
        assert self.P is not None and self.Q is not None
        assert self.bu is not None and self.bi is not None

        u_idx, i_idx, r = self._build_training_arrays()
        n = len(r)
        order = np.arange(n, dtype=np.int32)
        rng = np.random.default_rng(self.cfg.seed)

        lr = self.cfg.lr
        reg = self.cfg.reg

        for epoch in range(1, self.cfg.n_epochs + 1):
            rng.shuffle(order)
            se = 0.0  # sum squared error

            for t in order:
                u = int(u_idx[t])
                i = int(i_idx[t])
                rating = float(r[t])

                # Текущее предсказание
                pred = self.mu + self.bu[u] + self.bi[i] + float(self.P[u].dot(self.Q[i]))
                err = rating - pred
                se += err * err

                # Обновления (SGD)
                bu_old = self.bu[u]
                bi_old = self.bi[i]
                Pu_old = self.P[u].copy()
                Qi_old = self.Q[i].copy()

                self.bu[u] += lr * (err - reg * bu_old)
                self.bi[i] += lr * (err - reg * bi_old)
                self.P[u] += lr * (err * Qi_old - reg * Pu_old)
                self.Q[i] += lr * (err * Pu_old - reg * Qi_old)

            rmse = float(np.sqrt(se / max(1, n)))
            print(f"  эпоха {epoch}/{self.cfg.n_epochs} — RMSE: {rmse:.4f}")

        print("Обучение завершено.")

    async def _ensure_built(self) -> None:
        """
        Гарантирует, что модель обучена.
        """
        if self._built:
            return
        async with self._build_lock:
            if not self._built:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.train_model)
                self._built = True

    def _infer_virtual_user_vector(self, user_ratings: Dict[int, float]) -> Optional[np.ndarray]:
        """
        Инференс вектора латентных факторов для виртуального пользователя:
        решаем ridge-задачу по его оценкам.
        Формула: u = (Q_I^T Q_I + λI)^(-1) Q_I^T (r_I - μ - b_i)
        """
        if self.Q is None or self.bi is None:
            return None

        rows_Q: List[np.ndarray] = []
        targets: List[float] = []

        # Соберём пары только по фильмам, известным модели
        for iid, rating in user_ratings.items():
            if iid not in self.iid2idx:
                continue
            j = self.iid2idx[iid]
            rows_Q.append(self.Q[j])
            targets.append(float(rating) - self.mu - float(self.bi[j]))

        if len(rows_Q) < self.cfg.min_ratings_to_infer:
            return None

        Q_I = np.stack(rows_Q, axis=0).astype(np.float32)   # (m, k)
        y = np.array(targets, dtype=np.float32)             # (m,)
        regI = (self.cfg.reg * np.eye(self.cfg.n_factors, dtype=np.float32))
        A = Q_I.T @ Q_I + regI                               # (k, k)
        b = Q_I.T @ y                                        # (k,)

        # Решение линейной системы
        try:
            u = np.linalg.solve(A, b).astype(np.float32)
        except np.linalg.LinAlgError:
            u = (np.linalg.pinv(A) @ b).astype(np.float32)

        return u  # (k,)

    def predict_rating(
        self,
        virtual_user_ratings: Dict[int, float],
        item_id: int
    ) -> Optional[float]:
        """
        Предсказание рейтинга виртуального пользователя для одного фильма.
        """
        if self.Q is None or self.bi is None:
            return None
        if item_id not in self.iid2idx:
            return None

        u_vec = self._infer_virtual_user_vector(virtual_user_ratings)
        if u_vec is None:
            return None

        j = self.iid2idx[item_id]
        pred = self.mu + float(self.bi[j]) + float(u_vec.dot(self.Q[j]))
        pred = max(self.cfg.clip_min, min(self.cfg.clip_max, pred))
        return pred

    async def generate_recommendations(
        self,
        virtual_user_ratings: Dict[int, float],
        num_recommendations: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Генерация рекомендаций для виртуального пользователя.
        """
        await self._ensure_built()

        # Вычислим u_vec один раз для эффективности
        u_vec = self._infer_virtual_user_vector(virtual_user_ratings)
        if u_vec is None:
            # Если не можем вывести вектор — вернем популярное
            watched = set(virtual_user_ratings.keys())
            popular = self.dp.get_top_popular_movies(num_recommendations * 2)
            result: List[Tuple[int, float]] = []
            for mid in popular:
                if mid not in watched:
                    result.append((mid, 0.0))
                if len(result) >= num_recommendations:
                    break
            return result

        # Кандидаты: популярные минус просмотренные
        popular = self.dp.get_top_popular_movies(self.top_k_candidates)
        watched = set(virtual_user_ratings.keys())
        candidate_ids = [m for m in popular if m not in watched and m in self.iid2idx]

        preds: List[Tuple[int, float]] = []
        for iid in candidate_ids:
            j = self.iid2idx[iid]
            score = self.mu + float(self.bi[j]) + float(u_vec.dot(self.Q[j]))
            score = max(self.cfg.clip_min, min(self.cfg.clip_max, score))
            if score > 3.0:
                preds.append((iid, score))

        if not preds:
            # Фолбэк — популярные
            result: List[Tuple[int, float]] = []
            for mid in popular:
                if mid not in watched:
                    result.append((mid, 0.0))
                if len(result) >= num_recommendations:
                    break
            return result

        preds.sort(key=lambda x: x[1], reverse=True)
        return preds[:num_recommendations]
