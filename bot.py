import asyncio
from json import loads, dumps
import re

from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, ReplyKeyboardMarkup, KeyboardButton
from peewee import fn
from aiogram.utils import executor


from settings import BOT_TOKEN


bot = Bot(token=BOT_TOKEN)
dispatcher = Dispatcher(bot=bot)


@dispatcher.message_handler(commands={"ping"})
async def ping(event: types.Message):
    await event.reply("I'm alive")


@dispatcher.message_handler(commands={"start", "restart"})
async def start_handler(event: types.Message):
    pass


@dispatcher.message_handler(content_types=["text"])
async def text_handler(event: types.Message):
    await event.reply("Send photo with car")


@dispatcher.message_handler(content_types=["text"])
async def photo_handler(event: types.Message):
    await event.reply("Идёт распознавание...")


async def main():

    try:
        await dispatcher.start_polling()
    finally:
        await bot.close()


if __name__ == "__main__":
    asyncio.run(main())
    executor.start_polling(dispatcher, skip_updates=True)
