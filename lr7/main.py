import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
import ollama
import nest_asyncio
from dotenv import load_dotenv
import os

load_dotenv()
nest_asyncio.apply()
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

TOKEN = os.environ['TOKEN']

user_ids = {}
context_memory = {}


async def start(update: Update, context) -> None:
    await update.message.reply_text('Привет! Я чат-бот. Чем могу помочь?')


async def handle_message(update: Update, context):
    user_id = update.effective_user.id
    if user_id not in user_ids:
        user_ids[user_id] = {'last_message': None, 'preferences': {}}
        context_memory[user_id] = []

    message_text = update.message.text
    context_messages = context_memory[user_id]

    context_messages.append({'role': 'user', 'content': message_text})
    context_memory[user_id] = context_messages[-8:]

    try:
        response = ollama.chat(model='llama3:latest', messages=context_memory[user_id])
        await update.message.reply_text(response['message']['content'])
    except Exception as e:
        logging.error(f"Error while getting response from ollama: {e}")
        await update.message.reply_text('Произошла ошибка, попробуйте позже.')



async def main() -> None:
    application = ApplicationBuilder().token(TOKEN).build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    await application.run_polling()


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
