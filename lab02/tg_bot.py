from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

import vocabulary.config
from vocabulary.models import Deck, Model
from vocabulary.ai_vocabulary_deck import get_ai_vocabulary_deck


FIRST_MODEL_NAME = "gemma3"
SECOND_MODEL_NAME = "gemma3:1b"

vocabulary.config.set_config()
MESSAGES = vocabulary.config.BOT_MESSAGES

bot = Bot(token=vocabulary.config.TG_BOT_API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot=bot, storage=storage)


class Form(StatesGroup):
    llm_model = State()
    topic = State()
    first_language = State()
    second_language = State()
    temperature = State()
    word_count = State()


@dp.message(CommandStart())
async def cmd_start(message: types.Message, state: FSMContext):
    await state.set_state(Form.llm_model)
    button_gemma1b = InlineKeyboardButton(text=FIRST_MODEL_NAME, callback_data=FIRST_MODEL_NAME)
    button_gemma4b = InlineKeyboardButton(text=SECOND_MODEL_NAME, callback_data=SECOND_MODEL_NAME)
    gemmas_markup = InlineKeyboardMarkup(inline_keyboard=[[button_gemma1b, button_gemma4b]])
    await message.answer(MESSAGES["choose_model"], reply_markup=gemmas_markup)


@dp.callback_query(Form.llm_model)
async def process_llm_model(call: types.CallbackQuery, state: FSMContext):
    await state.set_state(Form.topic)
    await state.update_data(model_name=call.data)
    await call.message.answer(MESSAGES["topic"])


@dp.message(Form.topic)
async def process_topic(message: types.Message, state: FSMContext):
    if len(message.text) > 20:
        await message.reply(MESSAGES["topic_error"])
        return
    await state.update_data(topic=message.text)
    await state.set_state(Form.first_language)
    await message.reply(MESSAGES["first_language_input"])


@dp.message(Form.first_language)
async def process_first_language(message: types.Message, state: FSMContext):
    if len(message.text) > 20:
        await message.reply(MESSAGES["language_error"])
        return
    await state.update_data(first_language=message.text)
    await state.set_state(Form.second_language)
    await message.reply(MESSAGES["second_language_input"])


@dp.message(Form.second_language)
async def process_second_language(message: types.Message, state: FSMContext):
    if len(message.text) > 20:
        await message.reply(MESSAGES["language_error"])
        return
    await state.update_data(second_language=message.text)
    await state.set_state(Form.temperature)
    await message.reply(MESSAGES["temperature_input"])


@dp.message(Form.temperature)
async def process_temperature(message: types.Message, state: FSMContext):
    try:
        temperature = float(message.text)
        if not (0.1 <= temperature < 1):
            raise ValueError()
    except ValueError:
        await message.reply(MESSAGES["temperature_error"])
        return
    await state.update_data(temperature=temperature)
    await state.set_state(Form.word_count)
    await message.reply(MESSAGES["word_count_input"])


@dp.message(Form.word_count)
async def process_word_count(message: types.Message, state: FSMContext):
    try:
        word_count = int(message.text)
        if not (1 <= word_count <= 10):
            raise ValueError()
    except ValueError:
        await message.reply(MESSAGES["word_count_error"])
        return
    await state.update_data(word_count=word_count)
    user_data = await state.get_data()

    summary = MESSAGES["summary"].format(
        model_name=user_data["model_name"],
        topic=user_data["topic"],
        first_language=user_data["first_language"],
        second_language=user_data["second_language"],
        temperature=user_data["temperature"],
        word_count=user_data["word_count"],
    )

    await message.reply(summary, parse_mode=ParseMode.MARKDOWN)

    model = Model(
        model_name=user_data["model_name"],
        model_temerature=user_data["temperature"],
    )
    deck = Deck(
        first_language=user_data["first_language"],
        second_language=user_data["second_language"],
        topic=user_data["topic"],
        word_count=user_data["word_count"],
    )
    response = await get_ai_vocabulary_deck(
        deck=deck,
        model=model,
    )

    await message.answer(response, parse_mode=ParseMode.MARKDOWN)


if __name__ == '__main__':
    dp.run_polling(bot)
