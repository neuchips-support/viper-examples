import time
import torch
import subprocess
import json
import os
import signal
import pickle
import whisper
import speech_recognition as sr
import atexit
import gc

import streamlit as st
import numpy as np
from typing import Any, List

from streamlit_lottie import st_lottie
from streamlit import _bottom
from typing import Any
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import rag_db_app
from rag_db_func import (
    list_rag_data_versions,
    get_rag_data,
)

from rag_db_operations import (
    loading_uae_model,
    loading_qwen_emb_model,
    qwen_emb_encode,
    is_qwen_model,
    emb_encode,
)

# For loading the config file
import llm_rag_config as config

try:
    from llama_cpp import Llama as LlamaCppPython
except (ImportError, ModuleNotFoundError) as e:
    print(f"Error importing Llama from llama_cpp: {e} , try install the llama_cpp_python wheel.")

try:
    from llama_cpp import neu_vss_c as vss
except (ImportError, ModuleNotFoundError) as e:
    print(f"Error importing neu_vss_c from llama_cpp: {e} , try install the llama_cpp_python wheel.")

from tps_calculator import TPSCalculator
from user_logger import log_action
from language import prompts, set_language, translate, translate_example
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--language', choices=['en', 'zh-CN', 'zh-TW'], default='en')
args = parser.parse_args()
language = args.language

set_language(language)

DEVICE_NAME = "Phoenix"
TOPK_NUM = 4
EXTRA_TOPK_NUM = 20

def get_input_scale():
    if is_qwen_model(st.session_state.model_name):
        return 0.00188
    else:
        return 0.0374

def get_weight_scale():
    return get_input_scale()

def get_output_scale():
    if is_qwen_model(st.session_state.model_name):
        return 0.00787
    else:
        return 2.6562

def example_generator():

    example_list = []

    if not (st.session_state.rag_on and st.session_state.rag_loaded):
        if st.session_state.model_name == "TAIDE":
            example_list = [
                "幫我規劃台北一日遊的行程",
                "請幫我總結台版晶片法案",
                "台灣保育類物種",
            ]
        else:
            example_list = [
                translate_example("Safety Precautions for mountain climbing"),
                translate_example(" Python codes for finding the prime"),
                translate_example(" Tell me an interesting fact about Egypt"),
            ]

    return example_list


def rag_process(prompt, corpus_emb, raw_text):
    # embedding the activation prompt
    prompt_emb = emb_encode(prompt)

    if st.session_state.rag_engine == DEVICE_NAME and hasattr(corpus_emb, "shape"):
        weight_n_dim = corpus_emb.shape[0]
        output_scores_padding_buf = weight_n_dim
        # Check if weight_n_dim is 4-byte aligned due to hw limit
        if output_scores_padding_buf % 4 != 0:
            # If not aligned, round up to the next multiple of 4
            output_scores_padding_buf = ((output_scores_padding_buf // 4) + 1) * 4

        # quan to int8
        prompt_emb = quant_to_int8(prompt_emb)
        # special sequance order for device input
        prompt_emb_arr = prompt_emb.view(-1, 16).flip(1).flatten().numpy()
        #################################################################
        output_scores = np.zeros((output_scores_padding_buf,), dtype=np.int8)

        # run VSS
        vss_calc = loading_neu_vss_model()
        if not vss_calc.run(prompt_emb_arr, output_scores):
            print(f"Error. Failed to run VSS via {DEVICE_NAME}.")
            return

        # change to tensor
        print("output_scores size:", len(output_scores))
        # Only get the weight_n_dim from output buffer.
        output_scores = output_scores[:weight_n_dim]
        # Convert the truncated numpy array to a PyTorch tensor
        dot_result = torch.from_numpy(output_scores)
    else:
        # CPU prompt and weight matmul
        dot_result = dot_score(corpus_emb, prompt_emb)

    print("dot_result shape: ", dot_result.shape)
    values, indices = torch.topk(dot_result, min(len(dot_result), TOPK_NUM + EXTRA_TOPK_NUM), dim=0)
    print(f"VSS result topk {TOPK_NUM} + extra {EXTRA_TOPK_NUM}): {values}")

    rag_ele = []
    for i in range(min(len(indices), TOPK_NUM)):
        if indices[i] < len(raw_text):
            rag_ele.append(raw_text[indices[i]])
        else:
            print(
                f"Warning: Index {indices[i]} is out of bounds for raw_text with length {len(raw_text)}"
            )
    return rag_ele


# CPU only, matrix matmul
def dot_score(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


################################################


def check_session_state():
    black_list = [
        "chat_history",
        "rag_corpus_emb",
        "rag_raw_text",
        "preserving_prompt",
        "neutorch_model",
        "llm_streamer",
        "llama_cpp_service",
        "stop_thread",
        "model_downloading",
        "uploaded_file",
        "model_precompiler_cache_path",
        "model_prompt_template",
        "model_preserving_prompt_template",
        "model_max_preserving_prompt_count",
        # We store the object return from st.file_uploader,
        # with linux OS, it can be printout without any issue,
        # but in Windows OS, once print it out, it will crash,
        # due to internal _file_urls attribute could cause serialization or deserialization issues.
        "rag_uploaded_files",
    ]

    # Handle partial matches, don't print it out due to the serialization or deserialization issues.
    excluded_prefixes = ["rag_file_uploader_"]

    def is_blacklisted(key):
        # Check if key matches exactly
        if key in black_list:
            return True
        # Check if key starts with any of the excluded prefixes
        for prefix in excluded_prefixes:
            if key.startswith(prefix):
                return True
        return False

    # Debug output for session state keys
    print("=" * 100)
    for key in list(st.session_state.keys()):
        if not is_blacklisted(key):
            print(f"key: {key}  =>  {st.session_state[key]}")
    print("=" * 100)


def clear_text(text):
    return text.replace("'", "'\\''")


def gen_prompt_with_template(
    template="", system_request="", user_prompt="", pre_prompt="", pre_response=""
):
    placeholders = {
        "[system_setting]": system_request,
        "[pre_prompt_setting]": pre_prompt,
        "[pre_response_setting]": pre_response,
        "[user_prompt_setting]": user_prompt,
    }

    final_prompt = []

    # If no template is provided, use default template
    if not template:
        template = [
            {"role": "system", "content": "[system_setting]"},
            {"role": "user", "content": "[user_prompt_setting]"},
        ]

    for entry in template:
        filled_entry = entry.copy()
        for placeholder, value in placeholders.items():
            if placeholder in filled_entry["content"] and value:
                filled_entry["content"] = filled_entry["content"].replace(
                    placeholder, value
                )
        final_prompt.append(filled_entry)

    return final_prompt


def shellquote(user_prompt, sys_prompt, rag_element=None):
    ########################################################
    # Start to gen the system and user prompt
    system_request = ""

    if sys_prompt != "":
        if st.session_state.model_name == "TAIDE":
            # '''for taide '''
            system_request = sys_prompt + "，只會用繁體中文回答問題，盡可能簡短地回答"
        elif is_qwen_model(st.session_state.model_name):
            system_request = sys_prompt + "，只会用简体中文回答问题，尽可能简短地回答。"
        else:
            # '''for llama '''
            # system_request = (
            #     sys_prompt + ", aiming to keep your answers as brief as possible ."
            # )
            system_request = sys_prompt + prompts[language]

    else:
        if st.session_state.model_name == "TAIDE":
            # '''for taide '''
            system_request = "你是一個來自台灣的AI助理，你的名字是 TAIDE，樂於以台灣人的立場幫助使用者，只會用繁體中文回答問題，盡可能簡短地回答"
        elif is_qwen_model(st.session_state.model_name):
            system_request = "你是一个来自中国的AI助理，你的名字是Qwen，乐于以中国人的立场帮助用户，只会用简体中文回答问题，尽可能简短地回答。/no_think "# no think content
        else:
            # '''for llama '''
            # system_request = (
            #     "You are a helpful AI assistant, aiming to keep your answers in brief."
            # )
            system_request = sys_prompt + prompts[language]


    if rag_element:

        context = ""

        for index, value in enumerate(rag_element):
            # print(index, " : ", value)
            context = context + value + "\n"

        if st.session_state.model_name == "TAIDE":
            system_request = (
                system_request
                + "。使用以下上下文來回答使用者的問題。如果你不知道答案，就說你不知道，不要試圖編造答案。"
            )
            user_prompt = (
                "Context: "
                + context
                + "Question: "
                + user_prompt
                + " 只回覆有用的答案，不回覆任何其他內容。"
            )
        elif is_qwen_model(st.session_state.model_name):
            system_request = (
                system_request
                + "使用以下上下文来回答用户的问题。如果你不知道答案，就说你不知道，不要试图编造答案。"
            )
            user_prompt = (
                "Context: "
                + context
                + "Question: "
                + user_prompt
                + " 只回复有用的答案，不回复任何其他内容。"
            )
        else:
            system_request = (
                system_request
                + " Use the following Context to answer the Question. If you don't know the answer, just say that you need more information."
            )
            user_prompt = (
                "Context: "
                + context
                + "Question: "
                + user_prompt
                + "."
                + " Only return the helpful answer below and nothing else.\n"
            )

    system_request = system_request.replace("'", "'\\''")
    user_prompt = user_prompt.replace("'", "'\\''")
    ########################################################
    # Start to gen the complete prompt with template
    if (
        len(st.session_state.preserving_prompt) > 0
        and st.session_state.rag_loaded == False
        and st.session_state.model_preserving_prompt_template
    ):
        # Apply preserving_prompt
        pre_prompt = st.session_state.preserving_prompt[0]["prompt"].replace(
            "'", "'\\''"
        )
        pre_response = st.session_state.preserving_prompt[0]["content"].replace(
            "'", "'\\''"
        )
        final_prompt = gen_prompt_with_template(
            st.session_state.model_preserving_prompt_template,
            system_request,
            user_prompt,
            pre_prompt,
            pre_response,
        )
    else:
        final_prompt = gen_prompt_with_template(
            st.session_state.model_prompt_template, system_request, user_prompt
        )
    ########################################################
    return final_prompt


def session_state_init():
    if "cleanup_registered" not in st.session_state:
        st.session_state.cleanup_registered = False

    if "current_page" not in st.session_state:
        st.session_state.current_page = "rag_main_page"

    if "model_name" not in st.session_state:
        st.session_state.model_name = ""

    if "model_downloading" not in st.session_state:
        st.session_state.model_downloading = False

    if "model_path" not in st.session_state:
        st.session_state.model_path = ""

    if "model_precompiler_cache_path" not in st.session_state:
        st.session_state.model_precompiler_cache_path = ""

    if "model_prompt_template" not in st.session_state:
        st.session_state.model_prompt_template = ""

    if "model_preserving_prompt_template" not in st.session_state:
        st.session_state.model_preserving_prompt_template = ""

    if "model_max_preserving_prompt_count" not in st.session_state:
        st.session_state.model_max_preserving_prompt_count = (
            0 if config.max_preserving_prompt_count == 0 else 1
        )

    if "llama_cpp_service" not in st.session_state:
        st.session_state.llama_cpp_service = None

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "llm_running_state" not in st.session_state:
        st.session_state.llm_running_state = False

    if "rag_on" not in st.session_state:
        st.session_state.rag_on = False

    if "rag_toggle" not in st.session_state:
        st.session_state.rag_toggle = st.session_state.rag_on
        rag_enabled()

    if "rag_db" not in st.session_state:
        st.session_state.rag_db = ""

    if "rag_data_db_id" not in st.session_state:
        st.session_state.rag_data_db_id = None

    if "preserving_prompt" not in st.session_state:
        st.session_state.preserving_prompt = []

    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = ""

    if "rag_loaded" not in st.session_state:
        st.session_state.rag_loaded = False

    if "rag_corpus_emb" not in st.session_state:
        st.session_state.rag_corpus_emb = []

    if "rag_raw_text" not in st.session_state:
        st.session_state.rag_raw_text = []

    if "llm_engine" not in st.session_state:
        st.session_state.llm_engine = ""

    if "model_downloaded" not in st.session_state:
        st.session_state.model_downloaded = False

    if "rec_prompt" not in st.session_state:
        st.session_state.rec_prompt = ""

    if "valid_rec_prompt" not in st.session_state:
        st.session_state.valid_rec_prompt = False

    if "example_prompt_pressed" not in st.session_state:
        st.session_state.example_prompt_pressed = ""

    if "neutorch_model" not in st.session_state:
        st.session_state.neutorch_model = None

    if "llm_streamer" not in st.session_state:
        st.session_state.llm_streamer = None

    if "stop_thread" not in st.session_state:
        st.session_state.stop_thread = None

    if "llm_require_inference" not in st.session_state:
        st.session_state.llm_require_inference = None

    if "language" not in st.session_state:
        st.session_state.language = language

def register_cleanup_func():
    if st.session_state.cleanup_registered is not True:
        st.session_state.cleanup_registered = True
        atexit.register(lambda: cleanup_cache())

# Function to handle the Stop button click
def stop_btn_click_cb():
    if st.session_state.llm_running_state:
        st.session_state.llm_running_state = False
        st.session_state.valid_rec_prompt = False
        st.session_state.example_prompt_pressed = ""
        del st.session_state["llm_require_inference"]

    stop_btn_container.empty()
    return


def clear_btn_click_cb():
    st.session_state.chat_history = []
    st.session_state.preserving_prompt = []
    return


def record_btn_click_cb(recognizer, whisper_model):
    log_action("whisper recording.")
    AUDIO_FILE = "test.wav"
    # activate mic to recording
    with sr.Microphone() as source:
        # noise calibration
        # r.adjust_for_ambient_noise(source)
        print("Please Say something~")
        audio = recognizer.listen(source, phrase_time_limit=15)
        print("Recording completed!")

    # save to file
    with open("./" + AUDIO_FILE, "wb") as file:
        file.write(audio.get_wav_data())
        file.close()

    print("===========================================")
    st.session_state.rec_prompt = whisper_model.transcribe("test.wav")["text"]
    log_action("whisper prompt: "+ str(st.session_state.rec_prompt))

    # set this as the last step
    st.session_state.valid_rec_prompt = True
    return

@st.cache_resource
def loading_whisper_model():
    return whisper.load_model("base")


@st.cache_resource
def loading_recognizer():
    return sr.Recognizer()


@st.cache_resource
def loading_neu_vss_model():
    vss_calc = vss.PyNeuVssCalculator()
    print("VSS version: ", vss_calc.get_version())
    return vss_calc


@st.cache_resource
def loading_tps_calculator():
    tps = TPSCalculator()
    return tps


#### SDK refine start
# For loading the config value and apply the value
def get_supported_rag_lists(config):
    support_model_list = list(config.keys())
    support_engine_list = list(next(iter(config.values())).keys())
    return support_model_list, support_engine_list


def get_supported_model_lists(config):
    support_model_list = list(config.keys())
    return support_model_list


def get_supported_engine_lists(config, model):
    # Collect engines in order of appearance in the configuration
    if model not in config:
        return []  # Return an empty list if the model is not found

    support_engine_list = list(config[model].keys())
    return support_engine_list


# For the LLM model select and update the selected state
def update_model_state(model_list):
    llm_model_select = st.session_state.model_select
    if llm_model_select not in model_list:
        # Handle the case where llm_model_select is not in model_list
        st.error(translate("The selected model '{}' is not in the supported model list.").format(llm_model_select))
        return  False # Exit the function

    for model in model_list:
        if (
            llm_model_select == model
            and st.session_state.get("model_name") != llm_model_select
        ):
            reset_state()
            st.session_state.model_name = model
            # model changed reload config
            return  True
    # No need to reload config
    return False


def reset_state():
    print("!!!!!reset_state!!!!!")
    log_action("reset_state")
    st.session_state.chat_history = []
    st.session_state.preserving_prompt = []
    st.session_state.model_downloaded = False
    st.session_state.model_path = ""
    st.session_state.model_precompiler_cache_path = ""
    st.session_state.model_prompt_template = ""
    st.session_state.model_preserving_prompt_template = ""
    st.session_state.model_chat_format = ""
    st.session_state.system_prompt = ""
    st.session_state.example_prompt_pressed = ""
    # release the llama cpp python
    if st.session_state.get("llama_cpp_service"):
        del st.session_state["llama_cpp_service"]


# ==============================llama cpp python start =====================
def cleanup_llm(llm):
    print("🧹" "Clean up resources (llm)..")
    llm.close()
    del llm
    gc.collect()


def cleanup_cache():
    print("🧹" "Clean up resources (cache)..")
    st.cache_resource.clear()


def init_llm_cpp_python(
    model_math="", model_weight_cache_path="", chat_format=""
):

    if not model_math:
        raise ValueError("The model path cannot be empty. Please provide a valid path.")

    if not os.path.exists(model_math):
        raise FileNotFoundError(
            f"The specified model path does not exist: {model_math}"
        )

    with st.spinner(
        "Downloading " + st.session_state.model_name + " weight to device ..."
    ):
        log_action("Downloading " + st.session_state.model_name + " weight to device ...")
        print("model_math:", model_math)
        print("chat_format:", chat_format)

        start_time = time.time()
        llm_cpp_python = LlamaCppPython(
            model_path=model_math,
            n3k_id=0,
            neutorch_cache_path=model_weight_cache_path,
            chat_format=chat_format if chat_format else None,
            n_ctx=4096,
        )

        # store the llama cpp python service
        st.session_state.llama_cpp_service = llm_cpp_python
        # store the tokenizer and neutorch_model

        st.session_state.model_downloaded = True
        loading_time = time.time() - start_time
        print("loading time:",loading_time, "\n")
        log_action("loading time:" + str(loading_time))

        # start the tps count
        tps_cal = loading_tps_calculator()
        tps_cal.reset()
        tps_cal.set_tokenizer(llm_cpp_python.tokenizer())

        atexit.register(lambda: cleanup_llm(llm_cpp_python))


# llama-cpp-python
def start_llm_cpp_inference_stream(final_prompt):
    llm_cpp_python = st.session_state.llama_cpp_service

    if not llm_cpp_python:
        raise ValueError("Please init llama cpp python fist.")

    print("start_llm_cpp_inference_stream user_final_prompt: \n",final_prompt)
    log_action("start_llm_cpp_inference_stream. final_prompt: \n" + str(final_prompt))
    stream_response = llm_cpp_python.create_chat_completion(
        messages= final_prompt,
        stream=True
    )

    return stream_response


def llm_cpp_consume_output_stream(stream_response, output):
    output_text = ""
    for chunk in stream_response:
        # when stop_btn_click_cb been trigger
        if not st.session_state.llm_running_state:
            print("llama cpp python stop generate.")
            log_action("llama cpp python stop generate.")
            break

        token = chunk["choices"][0]["delta"].get("content", "")
        #print(token, end="", flush=True)
        if is_qwen_model(st.session_state.model_name) and (token == "<think>" or token == "</think>"):
            continue;# ignore the think tags
        output_text += token
        output.write(output_text)  # Update the Streamlit container

    return output_text
# ==============================llama cpp python end =====================


def update_engine_state():
    llm_engine_select = DEVICE_NAME # st.session_state.llm_engine_select
    if llm_engine_select != st.session_state.llm_engine:
        reset_state()
        st.session_state.llm_engine = llm_engine_select
        print("llm_engine change to : ", st.session_state.llm_engine)
        log_action("llm_engine change to : " + st.session_state.llm_engine)
        return True

    return False


# Based on the selected Engine get the model config
def get_model_config(model_name, config):
    if model_name not in config:
        st.error(translate("The model '{}' is not in the configuration.").format(model_name), icon="🚨")
        return None

    model_config = config[model_name]

    return {
        "model_path": model_config.get("path",""),
        "model_precompiler_cache_path": model_config.get("precompiler_cache_path", ""),
        "model_prompt_template": model_config.get("prompt_template", ""),
        "model_preserving_prompt_template": model_config.get("preserving_prompt_template", ""),
        "model_chat_format": model_config.get("chat_format", ""),
    }


# RAG enabler
def rag_enabled():
    if st.session_state.rag_on != st.session_state.rag_toggle:
        print("clear rag")
        st.session_state.preserving_prompt = []
        st.session_state.example_prompt_pressed = ""
        st.session_state.rag_corpus_emb = []
        st.session_state.rag_raw_text = []
        st.session_state.rag_db = ""
        st.session_state.rag_engine = ""
        st.session_state.rag_loaded = False
        st.session_state.rag_data_db_id = None
        # update rag_on state
        st.session_state.rag_on = st.session_state.rag_toggle
        log_action("rag_enabled :" + str(st.session_state.rag_on))


def handle_rag_data_selected(rag_data_db_id):
    # Check if the selected ID has changed
    if st.session_state["rag_data_db_id"] != rag_data_db_id:
        print("rag_data_db_id change to :", rag_data_db_id)
        st.session_state["rag_data_db_id"] = rag_data_db_id
        log_action("rag_data_selected : " + str(st.session_state["rag_data_db_id"]))


def handle_rag_db_selection():
    if not st.session_state.rag_on:
        return

    selected_db = "customized"

    # Show the file select UI
    if st.button(translate("Manage the Rag Data")):
        st.session_state["current_page"] = "rag_db_sub_page"
        log_action("Go to Manage the Rag Data page.")
        st.rerun()

    rag_datas = list_rag_data_versions()
    if rag_datas:
        rag_data_choices = {
            rag_data["version_name"]: rag_data["id"] for rag_data in rag_datas
        }
        rag_data_selected = st.selectbox(
            translate("Select Rag Data"), options=list(rag_data_choices.keys())
        )
        selected_rag_data_id = rag_data_choices[rag_data_selected]
        handle_rag_data_selected(selected_rag_data_id)
    else:
        # Reset the rag_data_db_id to None if no rag data in the DB
        st.session_state["rag_data_db_id"] = None

    if selected_db == st.session_state.rag_db:
        # rag_db not change
        return

    # rag_db change
    st.session_state.rag_loaded = False
    st.session_state.rag_db = selected_db


def handle_rag_engine_selection():
    if not st.session_state.rag_on:
        return

    rag_engine_select = DEVICE_NAME # st.session_state.rag_engine_select
    if st.session_state.rag_engine == rag_engine_select:
        # rag_engine not change
        return

    log_action("rag_engine_selected : " + st.session_state.rag_engine)
    # rag_engine change
    st.session_state.rag_loaded = False

    if rag_engine_select == "CPU":
        st.session_state.rag_engine = "CPU"
    else:
        # Rag on device
        vss_calc = loading_neu_vss_model()
        device_list = vss_calc.get_available_device()
        print("device_list=", device_list, "\n")
        if len(device_list) == 0:
            st.error(translate("Error. The device {} is not available.").format(DEVICE_NAME), icon="🚨")
            st.session_state.rag_engine = "CPU"
            return
        else:
            # bind_device for RAG
            if not vss_calc.bind_device(device_list[0]):
                st.error(
                    "Error. Failed to bind device ",
                    device_list[0],
                    ". CPU only.",
                    icon="🚨",
                )
                st.session_state.rag_engine = "CPU"
                return

            st.session_state.rag_engine = rag_engine_select


def customize_db_loading_rag_weight():
    corpus_emb = []
    raw_text = []
    rag_data_version_name = None

    if st.session_state.rag_data_db_id:
        rag_data = get_rag_data(st.session_state.rag_data_db_id)
        if rag_data:
            rag_data_version_name = rag_data["version_name"]
            raw_text = json.loads(rag_data["raw_data"])
            corpus_emb_numpy = pickle.loads(rag_data["emb_data"])
            corpus_emb = torch.tensor(corpus_emb_numpy)

    # All failed to load the rag db.
    if not rag_data_version_name or corpus_emb == [] or raw_text == []:
        print("Fail to load the customize rag db. \n")
        st.session_state.rag_loaded = False
        return [], []

    # Construct file paths based on the Rag Data version name
    full_file_path = f"{config.rag_db_path}{rag_data_version_name}/"
    full_file_prefix = f"{full_file_path}{rag_data_version_name}"
    os.makedirs(os.path.dirname(full_file_prefix), exist_ok=True)

    # Handle neutorch device weight loading for VSS
    if st.session_state.rag_engine == DEVICE_NAME:
        load_rag_weight(full_file_path, full_file_prefix, corpus_emb)
    else:
        # CPU mode don't have to load to card.
        st.session_state.rag_loaded = True
        log_action("CPU loading weight.")

    print("corpus_emb shape:", corpus_emb.shape)
    return corpus_emb, raw_text


def convertTensortoBin(full_file_prefix, corpus_emb):
    # Ensure the embedding is a tensor
    if not isinstance(corpus_emb, torch.Tensor):
        raise ValueError("The generated embedding is not a PyTorch tensor.")

    # quan to int8
    corpus_emb = quant_to_int8(corpus_emb)
    corpus_emb_numpy = corpus_emb.numpy()
    # Save the NumPy array directly to a .bin file
    corpus_emb_numpy.tofile(full_file_prefix + config.rag_db_bin_file_name)
    return full_file_prefix + config.rag_db_bin_file_name


def rag_convert_db(full_file_path, db_bin_path, input_dim):
    if st.session_state.rag_engine == "CPU":
        return None

    vss_calc = loading_neu_vss_model()
    # Check if the db_bin_path exists and is a .bin file
    if os.path.exists(db_bin_path):
        if not db_bin_path.endswith(".bin"):
            st.error(
                translate("The file '{}' exists but is not a .bin file.").format(db_bin_path), icon="🚨"
            )
            return None

    else:
        st.error(translate("The file '{}' not exist.").format(db_bin_path), icon="🚨")
        return None

    if not vss_calc.convert_db(
        input_dim, db_bin_path, full_file_path + config.weight_bin_file_dir
    ):
        st.error(translate("Error. Failed to convert VSS db to weight files."), icon="🚨")
        return None
    log_action("rag_convert_db:" + str(full_file_path))
    return full_file_path + config.weight_bin_file_dir


def load_rag_weight(full_file_path, full_file_prefix, corpus_emb):
    if st.session_state.rag_engine == "CPU" or not hasattr(corpus_emb, "shape"):
        return

    weight_k_dim = corpus_emb.shape[1]
    db_bin_path = convertTensortoBin(full_file_prefix, corpus_emb)
    # Check if the db_bin_path exists and is a .bin file
    weight_bin_file_dir_path = rag_convert_db(
        full_file_path, db_bin_path, weight_k_dim
    )

    if weight_bin_file_dir_path == None:
        return

    # Initialize
    q_scale = quant_scale()
    vss_calc = loading_neu_vss_model()
    if not vss_calc.initialize(weight_k_dim, weight_bin_file_dir_path, q_scale):
        st.error(translate("Error. Failed to initialize VSS."), icon="🚨")
        return

    st.session_state.rag_loaded = True
    log_action("load_rag_weight : " + str(st.session_state.rag_loaded))


def prepare_rag_database():
    return customize_db_loading_rag_weight()


# trigger vss rag and generate rag-prompt
def trigger_rag_and_gen_prompt(prompt):
    if (
        st.session_state.rag_on
        and st.session_state.rag_loaded
        and st.session_state.rag_corpus_emb != []
        and st.session_state.rag_raw_text != []
    ):
        # prepare vss rag and generate the prompt
        print("rag on!")
        with st.spinner("Generating rag prompt..."):
            log_action("Generating rag prompt.")
            time.sleep(0.5)  # for UI render the spinner
            rag_element = rag_process(
                prompt, st.session_state.rag_corpus_emb, st.session_state.rag_raw_text
            )

        print("size of rag_element: ", len(rag_element))
        log_action("size of rag_element: " + str(len(rag_element)))
        for index, value in enumerate(rag_element):
            print(index, " : ", value)
            print("--" * 100)
        word_esc = shellquote(prompt, st.session_state.system_prompt, rag_element)
    else:
        word_esc = shellquote(prompt, st.session_state.system_prompt)

    return word_esc


def llm_generate_wrapper(stop_event, llm_model, generation_kwargs):
    try:
        while not stop_event.is_set():
            llm_model.generate(**generation_kwargs)
            break
    except Exception as e:
        print(f"An error occurred during generation: {e}")


# quantization to int8 # scale = 0.0374
def quant_to_int8(prompt):
    prompt = torch.clamp(torch.round(prompt / get_input_scale()), -127, 127).to(torch.int8)

    return prompt


# special design for int8 quantization
@st.cache_resource
def quant_scale():
    sa = torch.tensor(get_input_scale(), dtype=torch.bfloat16)
    sw = torch.tensor(get_weight_scale(), dtype=torch.bfloat16)
    so = torch.tensor(get_output_scale(), dtype=torch.bfloat16)

    result = (sa * sw) / so

    # Convert the result to a Python float for display (optional)
    result_float = result.item()
    return result_float

# start the generic llm inference
def llm_start_inference(word_esc):
    if not st.session_state.get("llm_running_state", False):
        print("final prompt: ", word_esc)
        log_action("final prompt: " + str(word_esc))
        print("!!!!Start the LLM !!!!")
        llm_streamer = start_llm_cpp_inference_stream(word_esc)
        st.session_state.llm_streamer = llm_streamer
        st.session_state.llm_running_state = True
        st.rerun()

# get the generic llm inference response
def llm_receive_response():
    if st.session_state.get("llm_running_state", False):
        with st.chat_message("assistant", avatar="❇️"):
            lottie_container = st.empty()
            with lottie_container:
                with open("./resources/UI_elements/processing_animation.json", "r") as f:
                    data = json.load(f)

                st.markdown(
                    """
                        <style>
                            iframe {
                                justify-content: center;
                                display: block;
                                border-style:none;
                            }
                        </style>
                        """,
                    unsafe_allow_html=True,
                )
                st_lottie(data, height=80, key="gen")
                output_container = st.empty()
                response = llm_cpp_consume_output_stream(st.session_state.llm_streamer, output_container)
                log_action("inference response: " + str(response))
                stop_btn_container.empty()

            lottie_container.empty()
            return response

    return None


def llm_store_response_and_clean_state(response):
    if st.session_state.llm_require_inference and response:
        prompt = st.session_state.get("llm_require_inference")
        print("final response:" , response)
        # TPS Calculation
        tps_cal = loading_tps_calculator()
        tps_cal.set_output_response(response)
        tps_cal.calculate_tps()

        # Manage Chat History
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )
        # Manage Preserving Prompts
        st.session_state.preserving_prompt.append(
            {"prompt": prompt, "content": response}
        )
        # Check if preserving_prompt exceeds the max size and remove the oldest entry if necessary
        if (
            len(st.session_state.preserving_prompt)
            > st.session_state.model_max_preserving_prompt_count
        ):
            del st.session_state.preserving_prompt[0]

        # Reset States
        st.session_state.llm_running_state = False
        st.session_state.valid_rec_prompt = False
        st.session_state.example_prompt_pressed = ""
        del st.session_state["llm_streamer"]
        del st.session_state["stop_thread"]
        del st.session_state["llm_require_inference"]
        st.rerun()



#####################################################
if __name__ == "__main__":

    session_state_init()
    check_session_state()
    register_cleanup_func()

    if st.session_state["current_page"] == "rag_db_sub_page":
        rag_db_app.display_rag_db_app()
    else:
        st.markdown(
            f"<h1 style='text-align: center; color: white;'>RAG-LLM All on {DEVICE_NAME}</h1>",
            unsafe_allow_html=True,
        )

        whisper_model = loading_whisper_model()
        recognizer = loading_recognizer()
        recognizer.energy_threshold = 3000
        loading_uae_model()
        loading_qwen_emb_model()

        # Get the absolute path of the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))

        st.markdown(
            """
            <style>
                [data-testid=stSidebar] [data-testid=stImage]{
                    text-align: center;
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                    width: 100%;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Conditionally apply sidebar styles
        if (
            st.session_state["llm_running_state"]
            or st.session_state["model_downloading"]
        ):
            st.markdown(
                """
                <style>
                    [data-testid=stSidebar] {
                        pointer-events: none;  /* Disable interactions */
                        opacity: 0.5;          /* Grey out */
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <style>
                    [data-testid=stSidebar] {
                        pointer-events: auto;  /* Enable interactions */
                        opacity: 1;            /* Restore */
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )

        support_model_list = get_supported_model_lists(config.llm_config)
        with st.sidebar:
            # For the LLM model select and update the selected state
            image_path = os.path.join(script_dir, "resources/company_logo/Logo_NEUCHIPS_v_c+w.png")
            st.image(image_path, width=300)
            llm_model = st.selectbox(
                translate("Naming your Assistant"),
                support_model_list,
                key="model_select",
            )
            reload_config = update_model_state(support_model_list)

            # Based on the selected Engine get the path for loading the weight
            with st.container():
                reload_config = update_engine_state() or reload_config
                # Based on the selected Engine get the model config
                if reload_config:
                    model_config = get_model_config(
                        st.session_state.model_name,
                        config.llm_config,
                    )

                    if model_config:
                        # Update session state only if the value is not None
                        if model_config.get("model_path") == "":
                            st.error(
                                translate("Failed to load the model_path for'{}'.").format(st.session_state.model_name),
                                icon="🚨",
                            )

                        st.session_state.model_path = model_config.get("model_path")
                        st.session_state.model_precompiler_cache_path = model_config.get(
                            "model_precompiler_cache_path"
                        )
                        st.session_state.model_prompt_template = model_config.get(
                            "model_prompt_template"
                        )
                        st.session_state.model_preserving_prompt_template = model_config.get(
                            "model_preserving_prompt_template"
                        )
                        st.session_state.model_chat_format = model_config.get("model_chat_format")
                    else:
                        st.error(translate("Failed to load the model configuration."), icon="🚨")

                    log_action("reload_config. model_path: " + st.session_state.model_path)
                    log_action("reload_config. precompiler_cache_path: " + st.session_state.model_precompiler_cache_path)
                    log_action("reload_config. model_chat_format: " + st.session_state.model_chat_format)

                # For loading the weight
                if (
                    st.session_state.model_path != ""
                    and not st.session_state.model_downloading
                    and not st.session_state.model_downloaded
                ):
                    st.session_state.model_downloading = True
                    st.rerun()

            ########################################################
            # Naming your LLM
            @st.dialog(translate("Assistant Naming"))
            def char_naming():
                system_prompt = st.text_area(
                    translate("Naming your Assistant"), st.session_state.system_prompt
                )
                if st.button(translate("Submit")):
                    log_action("Assistant Naming: " + str(system_prompt))
                    if system_prompt:
                        st.session_state.system_prompt = system_prompt
                    st.rerun()

            with st.container():
                st.write(translate("Naming Your Own LLM Assistant"))
                if st.button(
                    translate("Assistant Naming"),
                    use_container_width=True,
                ):
                    char_naming()
            ########################################################
            st.toggle(
                translate("RAG Function"),
                key="rag_toggle",
                on_change=rag_enabled,
            )
            ########################################################
            with st.container(border=True):
                # Handle the selection for RAG
                #######################################################
                if st.session_state.rag_on:
                    check_session_state()

                handle_rag_db_selection()
                handle_rag_engine_selection()
                ########################################################
                # Prepare the rag database
                submitted = st.button(
                    translate("Submit"),
                    use_container_width=True,
                    disabled=(
                        not st.session_state.rag_on
                        or st.session_state.model_downloading
                        or st.session_state.rag_data_db_id is None
                    ),
                )
                if submitted:
                    log_action("RAG submitted.")
                    with st.spinner("Preparing RAG Database ~"):
                        st.session_state.rag_corpus_emb, st.session_state.rag_raw_text = (
                            prepare_rag_database()
                        )

            ########################################################
            if st.session_state.rag_on and st.session_state.rag_loaded:
                st.success(translate("RAG Database is Ready"), icon="✅")
                log_action("RAG Database is Ready.")
            ########################################################

        ########################################################
        # downloading weight here for block the UI(selectbox and radio) during downloading
        if st.session_state.model_downloading and not st.session_state.model_downloaded:
            init_llm_cpp_python(
                st.session_state.model_path,
                st.session_state.model_precompiler_cache_path,
                st.session_state.model_chat_format
            )

            st.session_state.model_downloading = False
            st.rerun()
        ########################################################

        # Display chat_history on app rerun
        for message in st.session_state.chat_history:
            with st.chat_message(
                message["role"], avatar="❇️" if message["role"] == "assistant" else "🧙‍♀️"
            ):
                st.markdown(message["content"])

        example_button_cols = _bottom.columns(3)
        disable_example_buttons = st.session_state.get("llm_running_state", False)

        # Example button
        example_prompts = example_generator()

        if (
            example_prompts != []
            and st.session_state.model_downloaded
        ):
            if example_button_cols[0].button(example_prompts[0], use_container_width=True, disabled=disable_example_buttons):
                st.session_state.example_prompt_pressed = example_prompts[0]
                log_action("example_button Click: " + str(st.session_state.example_prompt_pressed))

            if example_button_cols[1].button(example_prompts[1], use_container_width=True, disabled=disable_example_buttons):
                if example_prompts[1] == "請幫我總結台版晶片法案":
                    st.session_state.example_prompt_pressed = "請將這篇文章精簡條理化:「產業創新條例第10條之2及第72條條文修正案」俗稱「台版晶片法」,針對半導體、電動車、5G等技術創新且居國際供應鏈關鍵地位公司,提供最高25%營所稅投抵優惠,企業適用要件包含當年度研發費用、研發密度達一定規模,且有效稅率達一定比率。\
                                                                為因應經濟合作暨發展組織(OECD)國家最低稅負制調整,其中有效稅率門檻,民國112年訂為12%,113年料將提高至15%,但仍得審酌國際間最低稅負制實施情形。\
                                                                經濟部官員表示,已和財政部協商進入最後階段,除企業研發密度訂在6%,目前已確認,企業購置先進製程的設備投資金額達100億元以上可抵減。\
                                                                財政部官員表示,研商過程中,針對台灣產業與其在國際間類似的公司進行深入研究,在設備部分,畢竟適用產創10之2的業者是代表台灣隊打「國際盃」,投入金額不達100億元,可能也打不了。\
                                                                至於備受關注的研發費用門檻,經濟部官員表示,歷經與財政部來回密切討論,研發費用門檻有望落在60億至70億元之間。\
                                                                財政部官員指出,研發攸關台灣未來經濟成長動能,門檻不能「高不可攀」,起初雖設定在100億元,之所以會調降,正是盼讓企業覺得有辦法達得到門檻、進而適用租稅優惠,才有動力繼續投入研發,維持國際供應鏈關鍵地位。\
                                                                經濟部官員表示,因廠商研發費用平均為30、40億元,其中,IC設計業者介於30億至60億元範圍,若將門檻訂在100億元,符合條件的業者較少、刺激誘因不足;此外,若符合申請門檻的業者增加,將可提高企業在台投資金額,財政部稅收也能因此獲得挹注。\
                                                                IC設計業者近日頻頻針對產創10之2發聲,希望降低適用門檻,加上各國力拚供應鏈自主化、加碼補助半導體產業,經濟部官員表示,經濟部和財政部就產創10之2達成共識,爭取讓更多業者受惠,盼增強企業投資力道及鞏固台灣技術地位。\
                                                                財政部官員表示,租稅獎勵的制定必須「有為有守」,並以達到獎勵設置目的為最高原則,現階段在打「國內盃」的企業仍可適用產創第10條、10之1的租稅優惠,共同壯大台灣經濟發展。\
                                                                經濟部和財政部正就研發費用門檻做最後確認,待今明兩天預告子法之後,約有30天時間,可與業界進一步討論及調整,盼產創10之2能在6月上路。"
                else:
                    st.session_state.example_prompt_pressed = example_prompts[1]

                log_action("example_button Click: " +  str(st.session_state.example_prompt_pressed))

            if example_button_cols[2].button(example_prompts[2], use_container_width=True, disabled=disable_example_buttons):
                st.session_state.example_prompt_pressed = example_prompts[2]
                log_action("example_button Click: " +  str(st.session_state.example_prompt_pressed))
        ########################################################################

        # lower right corner side UI for handle the prompt
        cols = _bottom.columns([1.9, 17.2, 2.5, 2.6], gap="small")

        with cols[1]:
            prompt = st.chat_input(
                "What is up?",
                key="chat_in",
                disabled=(
                    not st.session_state.model_downloaded
                    or st.session_state.llm_running_state
                ),
            )

        # for whisper
        with cols[0]:
            record_btn = st.button(
                ":studio_microphone:",
                use_container_width=True,
                disabled=(
                    not st.session_state.model_downloaded
                    or st.session_state.llm_running_state
                ),
            )

        # for whisper
        if record_btn:
            record_msg_container = st.empty()
            with record_msg_container:
                st.markdown("recording~")
                record_btn_click_cb(recognizer, whisper_model)
                record_msg_container.empty()

        # for whisper
        if st.session_state.valid_rec_prompt:
            prompt = st.session_state.rec_prompt

        # for example_prompt
        if st.session_state.example_prompt_pressed != "":
            prompt = st.session_state.example_prompt_pressed

        if not st.session_state.llm_require_inference and prompt:
            st.session_state.llm_require_inference = prompt
            print("user prompt: ", prompt)
        #######################################################################
        # handle prompt start inference
        if st.session_state.get("llm_require_inference"):
            # show the user prompt
            with st.chat_message("user", avatar="🧙‍♀️"):
                st.markdown(st.session_state.llm_require_inference)

            if not st.session_state.get("llm_running_state", False):

                # generate final prompt
                log_action("User prompt: " + str(prompt))
                word_esc = trigger_rag_and_gen_prompt(prompt)

                # tps record
                tps_cal = loading_tps_calculator()
                tps_cal.reset()
                tps_cal.set_input_prompt(word_esc)
                tps_cal.set_start_time()

                # start the llm inference
                llm_start_inference(word_esc)

            # show stop button
            with cols[2]:
                stop_btn_container = st.empty()
                with stop_btn_container:
                    stop_btn = st.button(
                        "Stop",
                        on_click=stop_btn_click_cb,
                        key=f"abc",
                        use_container_width=True,
                    )

            # Handle LLM Response
            response = llm_receive_response()
            llm_store_response_and_clean_state(response)

        # Show the clear button
        if len(st.session_state.chat_history) != 0:
            with cols[3]:
                st.button(
                    "clear", on_click=clear_btn_click_cb, use_container_width=True
                )
