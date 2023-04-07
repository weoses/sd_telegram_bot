import io
import os
import threading
import time
from telebot import TeleBot
from telebot import types
import logging
import pathlib
import numpy
from modules import call_queue, shared, sd_samplers, scripts
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, \
    StableDiffusionProcessingImg2Img, process_images
from modules import txt2img

import importlib
from PIL import Image
from src import main, utils

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.DEBUG)

sout_h = logging.StreamHandler()
sout_h.setLevel(level=logging.DEBUG)
LOGGER.addHandler(sout_h)

opts = shared.opts
waiting_image_id = None

class SdTgBot:
    __slots__ = ('bot', 'running')

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
    
    def __finish_waiting(self, waiting:types.Message, supports_read_img, comment_data):
        msg = main.get_msg('telegram_bot_generated_msg', gen_data = comment_data)
        msg_send = msg
        msg_post = ''
        if len(msg) > 1024:
            msg_send = msg[:1023]
            msg_post = msg[1023:]
        
        self.bot.edit_message_media(
            message_id=waiting.message_id,
            chat_id=waiting.chat.id,
            media=types.InputMediaPhoto(supports_read_img, 
                                        caption=msg_send) )
        if msg_post:
            self.bot.send_message(chat_id=waiting.chat.id, text=msg_post)
        
    def __error_waiting(self, waiting:types.Message):
        self.bot.edit_message_media(
            message_id=waiting.message_id,
            chat_id=waiting.chat.id,
            media=None)

        self.bot.edit_message_text(message_id=waiting.message_id, 
                                   chat_id=waiting.chat.id, 
                                   text=main.get_msg('telegram_bot_generated_error_msg'))
        
    def __gen_processing(self, 
                       p:StableDiffusionProcessing,
                       waiting_update
                       ) -> Processed:

        LOGGER.debug(f'Gen {p.__class__} {p.prompt}, {p.width}x{p.height}')

        processed:Processed = None

        def run_gen():
            nonlocal processed
            shared.state.begin()
            try:
                processed = scripts.scripts_img2img.run(p, *p.script_args)
                if processed is None:
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

    def __get_arg_img_id(self, message:types.Message):
        img = None
        if message.photo: 
            img = message.photo[-1].file_id

        if not img and message.document:
            img = message.document.file_id
        
        if not img and message.reply_to_message:
            msg_old = message.reply_to_message
            
            if msg_old.photo:
                img = msg_old.photo[-1].file_id

            if not img and msg_old.document:
                img = message.document.file_id
        return img

    def __fill_args(self, p: StableDiffusionProcessing):
        last_arg_index = 1
        for script in p.scripts.scripts:
            if last_arg_index < script.args_to:
                last_arg_index = script.args_to
        p.script_args = [None] * last_arg_index
        p.script_args[0] = 0


    def __controlnet_args(self, p: StableDiffusionProcessing, image:Image):
        try:
            external_code = importlib.import_module('extensions.sd-webui-controlnet.scripts.external_code', 'external_code')
            units = [
                external_code.ControlNetUnit(
                    model=main.get_conf('telegram_bot_img2img_controlnet_model'),
                    module=main.get_conf('telegram_bot_img2img_controlnet_module'),
                    processor_res=main.get_conf('telegram_bot_img2img_controlnet_processor_res'),
                    threshold_a=main.get_conf('telegram_bot_img2img_controlnet_threshold_a'),
                    threshold_b=main.get_conf('telegram_bot_img2img_controlnet_threshold_b'),
                    guess_mode=False,
                    image={'image': numpy.array(image), 'mask' : None}
                )
            ]

            external_code.update_cn_script_in_processing(p, units)
        except:
            pass


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

    def on_img2img(self, message:types.Message):
        prompt = utils.get_arg(message.text)
        if not prompt:
            prompt = main.get_conf("telegram_bot_img2img_default_prompt")
            

        img = self.__get_arg_img_id(message)

        if not img:
            self.bot.send_message(message.chat.id, main.get_msg('telegram_bot_invalid_prompt_msg'))
            return
        
        file_props = self.bot.get_file(img)
        data = self.bot.download_file(file_props.file_path)

        input_data = io.BytesIO(data)

        img_pil = Image.open(input_data)

        LOGGER.debug(f"img2img incoming {img_pil.size[0]}x{img_pil.size[1]}")        
        # save img 
        # ex aspect radio 1024/512 = 0.5
        # need 128 x 128
        # x = 256
        # y = 128
        # y / aspect = x
        # 128 / 0.5 = 256
        xy = img_pil.size[0] / img_pil.size[1]
        if xy >= 1 : 
            # width  > height
            # incoming - 1000 x 500
            # need - 512x512
            # xy = 2
            # calc - 512 x 256  y = (x / xy)
            need_sizes = (int(main.get_conf('telegram_bot_img_width')), int(main.get_conf('telegram_bot_img_width') / xy))
        else :
            # width < height
            # incoming - 500 x 1000
            # need - 512x512
            # xy = 0.5
            # calc - 256 x 512 (y * xy)
            need_sizes = (int(main.get_conf('telegram_bot_img_height') * xy)), int(main.get_conf('telegram_bot_img_height'))

        img_pil = img_pil.resize(need_sizes)
        LOGGER.debug(f"img2img resizied {img_pil.size[0]}x{img_pil.size[1]}")
        img_pil.convert("RGB")

        def generate_call():
            p = StableDiffusionProcessingImg2Img(
                init_images=[img_pil],
                outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
                outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
                denoising_strength=main.get_conf("telegram_bot_img2img_denoising"),
                resize_mode=2,
                prompt=prompt,
                negative_prompt=main.get_conf('telegram_bot_negative_prompt'),
                sampler_name=main.get_conf('telegram_bot_sampler'),
                steps=int(main.get_conf('telegram_bot_steps')),
                cfg_scale=main.get_conf('telegram_bot_cfg_scale'),
                width=need_sizes[0],
                height=need_sizes[1])
            

            p.scripts = scripts.scripts_img2img
            self.__fill_args(p)
            
            if main.get_conf('telegram_bot_img2img_controlnet'):
                self.__controlnet_args(p, img_pil)

            waiting = self.__send_waiting(incoming=message)

            res = self.__gen_processing(
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

            gen_comment = ''
            if main.get_conf('telegram_bot_comment_send'):
                gen_comment = res.infotext(p, 0)

            self.__finish_waiting(waiting, output_data, gen_comment)

        call_queue.wrap_queued_call(generate_call)()
           
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
                sampler_name=main.get_conf('telegram_bot_sampler'),
                steps=int(main.get_conf('telegram_bot_steps')),
                cfg_scale=main.get_conf('telegram_bot_cfg_scale'),
                width=int(main.get_conf('telegram_bot_img_width')),
                height=int(main.get_conf('telegram_bot_img_height')),
            )

            p.scripts = scripts.scripts_txt2img
            self.__fill_args(p)
            
            waiting = self.__send_waiting(incoming=message)

            res = self.__gen_processing(
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
            
            gen_comment = ''
            if main.get_conf('telegram_bot_comment_send'):
                gen_comment = res.infotext(p, 0)

            self.__finish_waiting(waiting, output_data, gen_comment)

        call_queue.wrap_queued_call(generate_call)()

    def init_msgs(self):
        modes = main.get_conf('telegram_bot_commands')

        if 'start' in modes:
            self.bot.register_message_handler(
                callback=self.on_cmd_start,
                func=self.filter_msgs,
                commands=['start'])
        

        if 'text2img' in modes:
            self.bot.register_message_handler(
                callback=self.on_txt2img,
                func=self.filter_msgs,
                commands=[main.get_cmd("telegram_bot_text2img_cmd")])

        if 'img2img' in modes:    
            img2img_filter = '/'+main.get_cmd("telegram_bot_img2img_cmd")
            self.bot.register_message_handler(
                callback=self.on_img2img,
                func=lambda x : self.filter_msgs(x) 
                                and x.text 
                                and x.text.startswith(img2img_filter) )
    
    def start(self):
        
        while True:
            time.sleep(3) # TODO wait for polling stops
            try:
                self.running = True
                self.bot.polling()
            except Exception as e:
                LOGGER.warning("Telegram polling error - %s", e)
    
            if not self.running:
                return


    def stop(self):
        self.running = False
        self.bot.stop_polling()

    def __init__(self, token:str) -> None:
        self.bot = TeleBot(token=token)
        self.running = False
        self.init_msgs()


