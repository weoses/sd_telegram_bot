import threading
from modules import shared, devices, script_callbacks, processing, masking, images
from   src.telegram_bot import SdTgBot
import logging
import gradio as gr

bo = None
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.DEBUG)

sout_h = logging.StreamHandler()
sout_h.setLevel(level=logging.DEBUG)
LOGGER.addHandler(sout_h)

CONSTANT = {
    'telegram_bot_invalid_prompt_msg' : "Invalid prompt",
    'telegram_bot_generated_msg' : 'Generated!',
    'telegram_bot_generated_error_msg' : 'Error generating :(',
    'telegram_bot_start_msg' : 'Hello, i am Stable Diffusion bot',
    'telegram_bot_help_msg' : 'Some help',
    'telegram_bot_waiting_msg' : 'Generating, please wait',
    'telegram_bot_waiting_progress_msg' : 
                'Generating, please wait \n'
                'Current progress {progress} \n'
                'ETA {eta}',    
}

def create_bot():
    global bo
    try:
        token = shared.opts.data.get("telegram_bot_token")
        LOGGER.debug(f'Creating telegram bot')

        if token:
            bo = SdTgBot(token=token)
        else :
            bo = None
    except Exception as e:
        LOGGER.exception("Cant create telegram bot %s", e)        

def stop_bot():
    try:
        global bo
        if bo:
            LOGGER.debug(f'Stop telegram bot')
            bo.stop()
    except Exception as e:
        LOGGER.exception("Cant stop telegram bot %s", e)

def start_bot():
    try:
        global bo
        if bo:
            LOGGER.info(f'Start telegram bot')

            def safe_start():
                try:
                    bo.start()
                except Exception as e:
                    LOGGER.exception("Cant start telegram bot:  %s", e)       
                     
                    
            th = threading.Thread(target=safe_start) 
            th.daemon = True
            th.start()
    except Exception as e:
        LOGGER.exception("Cant start telegram bot:  %s", e)

def on_change_settings():
    try:
        stop_bot()
        create_bot()
        start_bot()
    except Exception as e:
        LOGGER.exception("Cant restart telegram bot %s", e)

def on_ui_settings():
    from src import main
    section = ('telegram_bot', "TelegramBot")

    shared.opts.add_option("telegram_bot_token", 
                           shared.OptionInfo('', 
                                             "Telegram bot token", 
                                             gr.Text, 
                                             section=section,
                                             onchange=main.on_change_settings))
    
    shared.opts.add_option("telegram_bot_autorized_chats", 
                           shared.OptionInfo('ALL', 
                                             "Autorized chat ids, separated by semicolon (;). Use 'ALL' for all chats", 
                                             gr.Text, 
                                             section=section))
    

    shared.opts.add_option("telegram_bot_negative_prompt", 
                           shared.OptionInfo('bad_person', 
                                             "Negative prompt", 
                                             gr.Text, 
                                             section=section))
    
    shared.opts.add_option("telegram_bot_steps", 
                           shared.OptionInfo(35, 
                                             "Steps", 
                                             gr.Slider,
                                             component_args={'step':1}, 
                                             section=section))
    
    shared.opts.add_option("telegram_bot_cfg_scale", 
                           shared.OptionInfo(7, 
                                             "CFG scale", 
                                             gr.Slider,
                                             component_args={'maximum':20}, 
                                             section=section))
    
    shared.opts.add_option("telegram_bot_img_width", 
                           shared.OptionInfo(512, 
                                             "Image width", 
                                             gr.Slider,
                                             component_args={'step':1, 'maximum':2000}, 
                                             section=section))
        
    shared.opts.add_option("telegram_bot_img_height", 
                           shared.OptionInfo(512, 
                                             "Image height", 
                                             gr.Slider,
                                             component_args={'step':1, 'maximum':2000}, 
                                             section=section))
    
    shared.opts.add_option("telegram_bot_text2img_cmd", 
                           shared.OptionInfo('txt2img', 
                                             "Slash command for txt2img", 
                                             section=section,
                                             onchange=main.on_change_settings))

def on_app_started(block, api):
    create_bot()
    start_bot()

def load():
    script_callbacks.on_ui_settings(on_ui_settings)
    script_callbacks.on_app_started(on_app_started)

def get_conf(code:str):
    if code in CONSTANT:
        return CONSTANT[code]
    
    return shared.opts.data.get(code)

def get_msg(code:str, **kwargs):
    strs = get_conf(code)

    if strs:
        return str.format(strs, **kwargs)
    
    return ''
