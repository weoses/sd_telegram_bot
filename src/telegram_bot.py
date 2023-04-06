import io
import os
import threading
import time
from telebot import TeleBot
from telebot import types
import logging
import pathlib
from modules import call_queue, shared, sd_samplers
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, \
    StableDiffusionProcessingImg2Img, process_images
from src import main, utils

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.DEBUG)

sout_h = logging.StreamHandler()
sout_h.setLevel(level=logging.DEBUG)
LOGGER.addHandler(sout_h)

opts = shared.opts
waiting_image_id = None

class SdTgBot:
    __slots__ = ('bot')

    def __send_waiting(self, incoming:types.Message):
        global waiting_image_id

        sent_msg = None
        if not waiting_image_id:
            loading = pathlib.Path(__file__) # ../../res/loading.png
            loading = pathlib.Path(loading.parent.parent, 'res', 'loading.png')
            with open(str(loading), "rb") as ph:
                sent_msg = self.bot.send_photo(
                    photo=ph, 
                    chat_id=incoming.chat.id, 
                    reply_to_message_id=incoming.id,
                    caption=main.get_msg("telegram_bot_waiting_msg"))
                if sent_msg.photo:
                    waiting_image_id = sent_msg.photo[0].file_id
        else:
            sent_msg = self.bot.send_photo(
                photo=waiting_image_id, 
                chat_id=incoming.chat.id, 
                reply_to_message_id=incoming.id,
                caption=main.get_msg("telegram_bot_waiting_msg"))

        return sent_msg
    
    def __update_waiting(self, waiting:types.Message, progress, eta):
        self.bot.edit_message_caption(
                message_id=waiting.message_id,
                chat_id=waiting.chat.id,
                caption=main.get_msg('telegram_bot_waiting_progress_msg', progress=progress, eta=eta))
    
    def __finish_waiting(self, waiting:types.Message, supports_read_img):
        self.bot.edit_message_media(
            message_id=waiting.message_id,
            chat_id=waiting.chat.id,
            media=types.InputMediaPhoto(supports_read_img, 
                                        caption=main.get_msg('telegram_bot_generated_msg')))
        
    def __error_waiting(self, waiting:types.Message):
        self.bot.edit_message_media(
            message_id=waiting.message_id,
            chat_id=waiting.chat.id)

        self.bot.edit_message_text(message_id=waiting.message_id, 
                                   chat_id=waiting.chat.id, 
                                   text=main.get_msg('telegram_bot_generated_error_msg'))
        
    def __gen_text2img(self, 
                       p:StableDiffusionProcessingTxt2Img,
                       waiting_update
                       ) -> Processed:

        LOGGER.debug(f'Gen text2img {p.prompt}, {p.width}x{p.height}')

        processed:Processed = None

        def run_gen():
            nonlocal processed
            shared.state.begin()
            try:
                processed = process_images(p)            
            finally:
                shared.state.end()
        
        th = threading.Thread(target=run_gen)
        th.start()

        while th.is_alive():
            time.sleep(3)

            eta = utils.get_eta()
            waiting_update(eta)

        return processed
        
    def filter_msgs(self, msg:types.Message):
        auth_chats = main.get_conf('telegram_bot_autorized_chats')
        if 'ALL' == auth_chats: return True
        if not auth_chats: return False

        return f'{msg.chat.id}' in auth_chats.split(';')
    
    def on_cmd_start(self, message:types.Message):
        self.bot.send_message(
            chat_id=message.chat.id,
            reply_to_message_id=message.id,
            text="Hello!")
        
    def on_txt2img(self, message:types.Message):
        prompt = utils.get_arg(message.text)
        if not prompt:
            self.bot.send_message(message.chat.id, main.get_msg('telegram_bot_invalid_prompt_msg'))
            return
        
        def generate_call():
            p = StableDiffusionProcessingTxt2Img(
                sd_model=shared.sd_model,
                outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
                outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
                prompt=prompt,
                negative_prompt=main.get_conf('telegram_bot_negative_prompt'),
                seed=-1,
                sampler_name=sd_samplers.samplers[0].name,
                steps=int(main.get_conf('telegram_bot_steps')),
                cfg_scale=main.get_conf('telegram_bot_cfg_scale'),
                width=int(main.get_conf('telegram_bot_img_width')),
                height=int(main.get_conf('telegram_bot_img_width')),
            )
            waiting = self.__send_waiting(incoming=message)

            res = self.__gen_text2img(
                p,
                lambda x: self.__update_waiting(waiting, x[0], x[1]))
            
            if not res:
                self.__error_waiting(message)
                return
            
            if len(res.images) == 0:
                self.__error_waiting(message)
                return
            
            img = res.images[0]
            output_data = io.BytesIO()
            img.save(output_data, format='jpeg')
            output_data.seek(0)
            self.__finish_waiting(waiting, output_data)

        call_queue.wrap_queued_call(generate_call)()

    def init_msgs(self):
        self.bot.register_message_handler(
            callback=self.on_cmd_start,
            func=self.filter_msgs,
            commands=['start'])
        
        self.bot.register_message_handler(
            callback=self.on_txt2img,
            func=self.filter_msgs,
            commands=['txt2img'])
        
    
    def start(self):
        self.bot.polling(non_stop=True)

    def stop(self):
        self.bot.stop_polling()

    def __init__(self, token:str) -> None:
        self.bot = TeleBot(token=token)
        self.init_msgs()


