import os
import torch
import gradio as gr
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from flores200_codes import flores_codes


def load_models():
    # build model and tokenizer
    model_name_dict = {
        "nllb-distilled-600M": "facebook/nllb-200-distilled-600M",
    }

    model_dict = {}

    for call_name, real_name in model_name_dict.items():
        print("\tLoading model: %s" % call_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(real_name)
        tokenizer = AutoTokenizer.from_pretrained(real_name)
        model_dict[call_name + "_model"] = model
        model_dict[call_name + "_tokenizer"] = tokenizer

    return model_dict


def translation(source, target, text):
    if len(model_dict) == 2:
        model_name = "nllb-distilled-600M"

    start_time = time.time()
    source = flores_codes[source]
    target = flores_codes[target]

    model = model_dict[model_name + "_model"]
    tokenizer = model_dict[model_name + "_tokenizer"]

    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=source,
        tgt_lang=target,
    )
    output = translator(text, max_length=400)

    end_time = time.time()

    output = output[0]["translation_text"]
    result = {
        "inference_time": end_time - start_time,
        "source": source,
        "target": target,
        "result": output,
    }
    return result


if __name__ == "__main__":
    print("\tinit models")

    global model_dict

    model_dict = load_models()

    # define gradio demo
    lang_codes = list(flores_codes.keys())
    inputs = [
        gr.inputs.Dropdown(lang_codes, default="English", label="Source"),
        gr.inputs.Dropdown(lang_codes, default="Nepali", label="Target"),
        gr.inputs.Textbox(lines=5, label="Input text"),
    ]

    outputs = gr.outputs.JSON()

    title = "The Master Betters Translator"

    demo_status = "App is running on CPU"
    desc = "This is the beta version of the master betters translator, which used the pre-trained model of facebook's no language left behind and fine-tuned with custom datasets. To use this app you need to have chosen the source and target language with your input text to get the output."
    description = (
        f"{desc} {demo_status}"
    )
    examples = [["English", "Nepali", "Hi. nice to meet you"]]

    gr.Interface(
        translation,
        inputs,
        outputs,
        title=title,
        description=description,
        examples=examples,
        examples_per_page=50,
    ).launch()
