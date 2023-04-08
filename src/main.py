from io import StringIO
import threading
import time
from modules import shared, devices, script_callbacks, processing, masking, images
from   src.telegram_bot import SdTgBot
import logging
import gradio as gr
import yaml
import importlib



LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.DEBUG)

sout_h = logging.StreamHandler()
sout_h.setLevel(level=logging.DEBUG)
LOGGER.addHandler(sout_h)

DEFAULT = {
    'telegram_bot_invalid_prompt_msg' : "Invalid prompt",
    'telegram_bot_generated_msg' : 'Generated!\n'
                                   '{gen_data}',
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
    'telegram_bot_sampler' : 'Euler a', 
    'telegram_bot_img2img_controlnet' : False,
    'telegram_bot_comment_send' : False,
    'telegram_bot_img2img_controlnet_processor_res' : 512,
    'telegram_bot_img2img_controlnet_threshold_a' : 100,
    'telegram_bot_img2img_controlnet_threshold_b' : 200,

    'telegram_bot_img2img_cmd': "img2img",
    'telegram_bot_text2img_cmd': "text2img",
}



overrides_msgs_obj = None
overrides_cmds_obj = None
restart_bot_event = threading.Event()
bot_instance = None
bot_thread = None

def create_bot_thread():
    global bot_thread
    global bot_instance
    
    if bot_thread:
        return
    
    def th():
        bot_finished = threading.Event()
        while True:
            try:            
                def start_bot():
                    global bot_instance
                    try:
                        bot_finished.clear()     
                        update_overrides()
                        token = shared.opts.data.get("telegram_bot_token")
                        LOGGER.debug(f'Creating telegram bot')

                        if token:
                            bot_instance = SdTgBot(token=token)
                        else:
                            bot_instance = None
                            bot_finished.set()
                            return
                    
                        LOGGER.info(f'Start telegram bot')
                        bot_instance.run()
                        bot_finished.set()
                    except Exception as e:
                        LOGGER.exception("Bot run exception %s", e)
                
                start_th = threading.Thread(target=start_bot)
                start_th.daemon = True
                start_th.start()
                
                restart_bot_event.wait()

                if bot_instance != None:
                    LOGGER.debug(f'Stop telegram bot')
                    bot_instance.stop()
                    bot_finished.wait()

                restart_bot_event.clear()
            except Exception as  e:
                LOGGER.exception("TelegramBot error: %s", e)

    bot_thread = threading.Thread(target=th)
    bot_thread.daemon = True
    bot_thread.start()
        
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
    restart_bot_event.set()

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
    
    samplers  = [x.name for x in shared.list_samplers()]
    shared.opts.add_option("telegram_bot_sampler", 
                           shared.OptionInfo(samplers[0], 
                                             "Sampler", 
                                             gr.Dropdown,
                                             component_args={"choices": samplers},
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
                                             "Image2image denoising strengh", 
                                             gr.Slider,
                                             component_args={'maximum':1}, 
                                             section=section))    

    shared.opts.add_option("telegram_bot_comment_send", 
                           shared.OptionInfo(False, 
                                             "Add generation data to imgs", 
                                             gr.Checkbox, 
                                             section=section))                                    
    
    try:
        external_code = importlib.import_module('extensions.sd-webui-controlnet.scripts.external_code', 'external_code')
        shared.opts.add_option("telegram_bot_img2img_controlnet", 
                           shared.OptionInfo(False, 
                                             "Use controlnet for img2img", 
                                             gr.Checkbox, 
                                             section=section))
        models = external_code.get_models()
        modules = external_code.get_modules()
        shared.opts.add_option("telegram_bot_img2img_controlnet_model", 
                           shared.OptionInfo(models[0], 
                                             "Controlnet model", 
                                             gr.Dropdown,
                                             component_args={"choices": models},
                                             section=section))
        
        shared.opts.add_option("telegram_bot_img2img_controlnet_module", 
                           shared.OptionInfo(modules[0], 
                                             "Controlnet module", 
                                             gr.Dropdown,
                                             component_args={"choices": modules},
                                             section=section))
        
        shared.opts.add_option("telegram_bot_img2img_controlnet_processor_res", 
                           shared.OptionInfo(512, 
                                             "Controlnet annotator resolution", 
                                             gr.Slider,
                                             component_args={'maximum':2048, 'step':1}, 
                                             section=section))    
        
        shared.opts.add_option("telegram_bot_img2img_controlnet_threshold_a", 
                           shared.OptionInfo(100, 
                                             "Controlnet threshold a (Canny low threshold)", 
                                             gr.Slider,
                                             component_args={'maximum':255, 'step' :1}, 
                                             section=section))    

        shared.opts.add_option("telegram_bot_img2img_controlnet_threshold_b", 
                           shared.OptionInfo(200, 
                                             "Controlnet threshold b (Canny hight threshold)", 
                                             gr.Slider,
                                             component_args={'maximum':255, 'step' :1}, 
                                             section=section))    
    except:
        pass
        

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
    create_bot_thread()

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

