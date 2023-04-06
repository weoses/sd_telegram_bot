from io import StringIO
import threading
from modules import shared, devices, script_callbacks, processing, masking, images
from   src.telegram_bot import SdTgBot
import logging
import gradio as gr
import yaml

bot_instance = None
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.DEBUG)

sout_h = logging.StreamHandler()
sout_h.setLevel(level=logging.DEBUG)
LOGGER.addHandler(sout_h)

DEFAULT = {
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

    'telegram_bot_img2img_default_prompt': 'anime',
    'telegram_bot_commands': ["start", "help", "text2img", "img2img"],
    'telegram_bot_steps' : 35, 


    'telegram_bot_img2img_cmd': "img2img",
    'telegram_bot_text2img_cmd': "text2img",
}


overrides_msgs_obj = None
overrides_cmds_obj = None

def create_bot():
    global bot_instance
    try:
        update_overrides()
        token = shared.opts.data.get("telegram_bot_token")
        LOGGER.debug(f'Creating telegram bot')

        if token:
            bot_instance = SdTgBot(token=token)
        else :
            bot_instance = None
    except Exception as e:
        LOGGER.exception("Cant create telegram bot %s", e)        

def stop_bot():
    try:
        global bot_instance
        if bot_instance:
            LOGGER.debug(f'Stop telegram bot')
            bot_instance.stop()
    except Exception as e:
        LOGGER.exception("Cant stop telegram bot %s", e)

def start_bot():
    try:
        global bot_instance
        if bot_instance:
            LOGGER.info(f'Start telegram bot')

            def safe_start():
                try:
                    bot_instance.start()
                except Exception as e:
                    LOGGER.exception("Cant start telegram bot:  %s", e)       
                     
                    
            th = threading.Thread(target=safe_start) 
            th.daemon = True
            th.start()
    except Exception as e:
        LOGGER.exception("Cant start telegram bot:  %s", e)

def update_overrides():
    global overrides_msgs_obj
    global overrides_cmds_obj
    overrides_msgs_obj = None
    overrides_cmds_obj = None

    try :
        overrides_str = shared.opts.data.get('telegram_bot_msgs')
        if overrides_str:
            overrides_msgs_obj = yaml.load(StringIO(overrides_str), Loader=yaml.SafeLoader)
    except Exception as e:
        LOGGER.exception("Cant load overrides for messages: %s", e)
    
    try :
        overrides_str = shared.opts.data.get('telegram_bot_cmds')
        if overrides_str:
            overrides_cmds_obj = yaml.load(StringIO(overrides_str), Loader=yaml.SafeLoader)
    except Exception as e:
        LOGGER.exception("Cant load overrides for commands: %s", e)

def on_change_settings():
    '''Change settings callback. Restart bot, reload messanges overrides'''
    try:
        stop_bot()
        create_bot()
        start_bot()
    except Exception as e:
        LOGGER.exception("Cant restart telegram bot %s", e)

def on_ui_settings():
    """ Ui create function """

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
    
    shared.opts.add_option("telegram_bot_commands", 
                           shared.OptionInfo(
                                            ["start", "help", "text2img", "img2img"], 
                                            "Bot commands", 
                                            gr.CheckboxGroup, 
                                            lambda: {"choices": ["start", "help", "text2img", "img2img"]},
                                            section=section,
                                            onchange=main.on_change_settings)),
    
    shared.opts.add_option("telegram_bot_img2img_default_prompt", 
                           shared.OptionInfo('anime', 
                                             "Img2img default positive prompt", 
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
    
    shared.opts.add_option("telegram_bot_img2img_denoising", 
                           shared.OptionInfo(0.68, 
                                             "Image2image denoising strenght", 
                                             gr.Slider,
                                             component_args={'maximum':1}, 
                                             section=section))
    
    shared.opts.add_option("telegram_bot_cmds", 
                           shared.OptionInfo('', 
                                             "Slash command names overrides (yml, key-value)", 
                                             section=section,
                                             onchange=main.on_change_settings))
    
    shared.opts.add_option("telegram_bot_msgs", 
                           shared.OptionInfo('', 
                                             "Bot messanges overrides (yml, key-value)", 
                                             section=section,
                                             onchange=main.on_change_settings))

def on_app_started(block, api):
    create_bot()
    start_bot()

def load():
    script_callbacks.on_ui_settings(on_ui_settings)
    script_callbacks.on_app_started(on_app_started)

def get_conf(code:str):
    data = shared.opts.data.get(code)
    if data is None:
        return DEFAULT[code]
    
    return data

def get_msg(code:str, **kwargs):
    """ Get message by code, check for overrides, format by kwargs"""
    global overrides_msgs_obj
    strs = None
    if overrides_msgs_obj:        
        if code in overrides_msgs_obj:
            strs = overrides_msgs_obj[code]

    if not strs:
        if code in DEFAULT:
            strs = DEFAULT[code]

    if strs:
        return str.format(strs, **kwargs)
    
    return ''

def get_cmd(code:str):
    """ Get cmd alias by code, check for overrides"""
    global overrides_cmds_obj
    strs = None
    if overrides_cmds_obj:        
        if code in overrides_cmds_obj:
            strs = overrides_cmds_obj[code]

    if not strs:
        if code in DEFAULT:
            strs = DEFAULT[code]

    return strs

