import asyncio
from json import loads, dumps
import re
from threading import Thread
from pathlib import Path

from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, ReplyKeyboardMarkup, KeyboardButton
from peewee import fn
from aiogram.utils import executor
from yolov5.detect import detect

from settings import BOT_TOKEN

WEIGHTS = Path("./weights.pt")
PHOTO_PATH = Path("./images")

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


@dispatcher.message_handler(content_types=["photo"])
async def photo_handler(message: types.Message):
    await message.reply("Идёт распознавание...")

    photo_unique_id = message.photo[-1].file_unique_id
    img_path = PHOTO_PATH / f'{photo_unique_id}.jpg'
    await message.photo[-1].download(str(img_path))
    thread = Thread(target=detect, kwargs={'weights': WEIGHTS,
                                           'source': str(img_path),
                                           'imgsz': 256,
                                           'max_det': 300,
                                           'save_crop': True})
    thread.start()
    thread.join()

    output_dir_path = Path('./runs/detect/')
    subdirs_count = len(list(output_dir_path.iterdir()))
    output_img_path = output_dir_path / f"exp{subdirs_count}" / img_path.name

    with open(f'./{output_img_path}', 'rb') as file:
        await message.reply_photo(photo=file)





async def main():
    try:
        await dispatcher.start_polling()
    finally:
        await bot.close()


if __name__ == "__main__":
    asyncio.run(main())
    executor.start_polling(dispatcher, skip_updates=True)
