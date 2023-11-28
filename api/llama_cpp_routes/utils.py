from api.models import GENERATE_ENGINE
from api.utils.request import llama_outer_lock, llama_inner_lock


def get_llama_cpp_engine():
    # NOTE: This double lock allows the currently streaming model to
    # check if any other requests are pending in the same thread and cancel
    # the stream if so.
    llama_outer_lock.acquire()
    release_outer_lock = True
    try:
        llama_inner_lock.acquire()
        try:
            llama_outer_lock.release()
            release_outer_lock = False
            yield GENERATE_ENGINE
        finally:
            llama_inner_lock.release()
    finally:
        if release_outer_lock:
            llama_outer_lock.release()
