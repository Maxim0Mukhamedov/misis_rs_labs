from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

import asyncio
import re
from typing import Dict, Optional, Tuple, List

import filtering.config
from filtering.data_handler import DataProcessor
from filtering.collaborative_filtering import CollaborativeFiltering
from filtering.matrix_factorization import MatrixFactorization, MFConfig


filtering.config.set_config()
BOT_MESSAGES = filtering.config.BOT_MESSAGES or {}

def _msg(key: str, default: str = "INTERNAL ERROR") -> str:
    return BOT_MESSAGES.get(key, default)

METHOD_CF = "COLLABORATIVE_FILTERING"
METHOD_MF = "MATRIX_FACTORIZATION"
METHOD_BOTH = "BOTH"
ACTION_ADD_RATING = "ADD_RATING"

button_cf = InlineKeyboardButton(text="Collaborative Filtering", callback_data=METHOD_CF)
button_mf = InlineKeyboardButton(text="Matrix Factorization", callback_data=METHOD_MF)
button_both = InlineKeyboardButton(text="Оба метода", callback_data=METHOD_BOTH)
button_add_rating = InlineKeyboardButton(text="➕ Добавить оценку", callback_data=ACTION_ADD_RATING)

def build_main_markup() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [button_cf],
            [button_mf],
            [button_both],
            [button_add_rating],
        ]
    )

bot = Bot(token=filtering.config.TG_BOT_API_TOKEN)
storage = MemoryStorage()
dispatcher = Dispatcher(bot=bot, storage=storage)


class Form(StatesGroup):
    ratings = State()
    choose_method = State()


USER_RATINGS: Dict[int, Dict[int, float]] = {}


_dp: Optional[DataProcessor] = None
_cf: Optional[CollaborativeFiltering] = None
_mf: Optional[MatrixFactorization] = None
_models_lock = asyncio.Lock()


async def _ensure_data():
    global _dp
    if _dp is None:
        _dp = DataProcessor()
        await _dp.load_data()


async def _ensure_models() -> Tuple[CollaborativeFiltering, MatrixFactorization]:
    global _dp, _cf, _mf
    async with _models_lock:
        await _ensure_data()
        if _cf is None:
            _cf = CollaborativeFiltering(data_processor=_dp, min_common_items=3, top_k=20)
        if _mf is None:
            _mf = MatrixFactorization(data_processor=_dp, config=MFConfig(), top_k_candidates=1000)
    return _cf, _mf


def _parse_rating_line(text: str) -> Optional[Tuple[int, int]]:
    """
    Ожидаем формат: "<movie_id> <rating>" или "<movie_id>,<rating>"
    rating — целое 1..5
    """
    s = text.strip()
    m = re.match(filtering.config.RATING_FORMAT, s)
    if not m:
        return None
    movie_id = int(m.group(1))
    rating = int(m.group(2))
    if rating < 1 or rating > 5:
        return None
    return movie_id, rating


def _ratings_count(chat_id: int) -> int:
    return len(USER_RATINGS.get(chat_id, {}))


def _ensure_user(chat_id: int) -> None:
    if chat_id not in USER_RATINGS:
        USER_RATINGS[chat_id] = {}


def _ratings_summary(dp: DataProcessor, ratings: Dict[int, float]) -> str:
    if not ratings:
        return "—"
    lines: List[str] = []
    for mid, r in ratings.items():
        title = dp.get_movie_title(mid) if _dp is not None else f"Фильм {mid}"
        lines.append(f"- {title} (id {mid}) — {r}")
    return "\n".join(lines)


@dispatcher.message(CommandStart())
async def cmd_start(message: types.Message, state: FSMContext):
    await _ensure_data()
    _ensure_user(message.chat.id)

    if _ratings_count(message.chat.id) < 3:
        await state.set_state(Form.ratings)
        await message.answer(
            _msg("intro_collect").format(num=filtering.config.NUM_RECOMMENDATION),
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[[button_add_rating]]),
        )
        return

    await state.set_state(Form.choose_method)
    await message.answer(_msg("choose_method"), reply_markup=build_main_markup())


@dispatcher.message(Command("random"))
async def cmd_random(message: types.Message, state: FSMContext):
    global _dp
    await _ensure_data()
    sample = _dp.get_random_movies(5)
    if not sample:
        await message.answer(_msg("random_empty"))
        return
    lines = []
    for mid in sample:
        title = _dp.get_movie_title(mid)
        lines.append(f"- {title} (id {mid})")
    await message.answer(_msg("random_header")+"\n".join(lines))


@dispatcher.message(Command("reset"))
async def cmd_reset(message: types.Message, state: FSMContext):
    USER_RATINGS[message.chat.id] = {}
    await state.set_state(Form.ratings)
    await message.answer(
        _msg("reset_done").format(num=filtering.config.NUM_RECOMMENDATION),
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[[button_add_rating]]),
    )


@dispatcher.message(Command("list"))
async def cmd_list(message: types.Message, state: FSMContext):
    await _ensure_data()
    ratings = USER_RATINGS.get(message.chat.id, {})
    if not ratings:
        await message.answer(_msg("list_empty"))
        return
    await message.answer(_msg("list_header"))


@dispatcher.callback_query()
async def universal_callback(call: types.CallbackQuery, state: FSMContext):
    """
    Обрабатываем все нажатия кнопок: выбор метода и добавление оценки.
    """
    global _dp
    await _ensure_data()
    choice = call.data or ""
    await call.answer()

    if choice == ACTION_ADD_RATING:
        await state.set_state(Form.ratings)
        await call.message.answer(_msg("add_rating_prompt"))
        return

    user_ratings = USER_RATINGS.get(call.message.chat.id, {})
    if len(user_ratings) < 3:
        await state.set_state(Form.ratings)
        await call.message.answer(_msg("need_three_first"))
        return

    cf, mf = await _ensure_models()
    num = filtering.config.NUM_RECOMMENDATION

    if choice == METHOD_CF:
        recs_cf = await cf.generate_recommendations(user_ratings, num_recommendations=num)
        text_cf = _format_recommendations(_dp, recs_cf, header=_msg("cf_header"))
        await call.message.answer(text_cf, parse_mode=ParseMode.HTML)

    elif choice == METHOD_MF:
        recs_mf = await mf.generate_recommendations(user_ratings, num_recommendations=num)
        text_mf = _format_recommendations(_dp, recs_mf, header=_msg("mf_header"))
        await call.message.answer(text_mf, parse_mode=ParseMode.HTML)

    elif choice == METHOD_BOTH:
        recs_cf = await cf.generate_recommendations(user_ratings, num_recommendations=num)
        recs_mf = await mf.generate_recommendations(user_ratings, num_recommendations=num)
        text_cf = _format_recommendations(_dp, recs_cf, header=_msg("cf_header"))
        text_mf = _format_recommendations(_dp, recs_mf, header=_msg("mf_header"))
        await call.message.answer(text_cf, parse_mode=ParseMode.HTML)
        await call.message.answer(text_mf, parse_mode=ParseMode.HTML)

    else:
        await call.message.answer(_msg("unknown_method"))
        return

    await state.set_state(Form.choose_method)
    await call.message.answer(
        _msg("choose_again"),
        reply_markup=build_main_markup()
    )


@dispatcher.message(Form.ratings)
async def process_rating_line(message: types.Message, state: FSMContext):
    global _dp
    await _ensure_data()
    _ensure_user(message.chat.id)

    parsed = _parse_rating_line(message.text or "")
    if not parsed:
        await message.reply(_msg("rating_parse_error"))
        return

    movie_id, rating = parsed

    if movie_id not in _dp.ratings_df["movie_id"].unique():
        await message.reply(_msg("unknown_movie"))
        return

    USER_RATINGS[message.chat.id][movie_id] = float(rating)
    title = _dp.get_movie_title(movie_id)
    await message.reply(_msg("saved_rating", "Сохранил: ") + f"{title} (id {movie_id}) — {rating}")

    cnt = _ratings_count(message.chat.id)
    num = filtering.config.NUM_RECOMMENDATION
    if cnt < num:
        await message.answer(_msg("need_more").format(num=num))
        return

    await state.set_state(Form.choose_method)
    await message.answer(
        _msg("choose_method"),
        reply_markup=build_main_markup()
    )


def _format_recommendations(dp: DataProcessor, recs: List[Tuple[int, float]], header: str) -> str:
    lines: List[str] = [header]
    for mid, score in recs:
        title = dp.get_movie_title(mid)
        lines.append(f"- {title} (id {mid}) — <b>{score:.2f}</b>")
    return "\n".join(lines)


if __name__ == "__main__":
    dispatcher.run_polling(bot)
