import modules.shared as shared
import time

def get_eta() -> tuple: 
    """(progress, eta)"""
    # ProgressApi copypaste 
    # avoid dividing zero
    if shared.state.job_count == 0:
        return (0,0)
    
    progress = 0.01

    if shared.state.job_count > 0:
        progress += shared.state.job_no / shared.state.job_count
    if shared.state.sampling_steps > 0:
        progress += 1 / shared.state.job_count * shared.state.sampling_step / shared.state.sampling_steps

    time_since_start = time.time() - shared.state.time_start
    eta = (time_since_start/progress)
    eta_relative = eta-time_since_start

    progress = min(progress, 1)
    return (progress, eta_relative)

def get_arg(msg_text:str) -> str:
    if msg_text:
        args = msg_text.split(" ", maxsplit=1)
        if len(args) == 2:
            return args[1]
    return None
