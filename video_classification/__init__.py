import os
import gradio as gr
from utils import (
    create_gif_from_video_file,
    download_youtube_video,
    get_num_total_frames,
)
from transformers import pipeline
from huggingface_hub import HfApi, ModelFilter

FRAME_SAMPLING_RATE = 4
DEFAULT_MODEL = "facebook/timesformer-base-finetuned-k400"

VALID_VIDEOCLASSIFICATION_MODELS = [
    "MCG-NJU/videomae-large-finetuned-kinetics",
    "facebook/timesformer-base-finetuned-k400",
    "fcakyon/timesformer-large-finetuned-k400",
    "MCG-NJU/videomae-base-finetuned-kinetics",
    "facebook/timesformer-base-finetuned-k600",
    "fcakyon/timesformer-large-finetuned-k600",
    "facebook/timesformer-hr-finetuned-k400",
    "facebook/timesformer-hr-finetuned-k600",
    "facebook/timesformer-base-finetuned-ssv2",
    "fcakyon/timesformer-large-finetuned-ssv2",
    "facebook/timesformer-hr-finetuned-ssv2",
    "MCG-NJU/videomae-base-finetuned-ssv2",
    "MCG-NJU/videomae-base-short-finetuned-kinetics",
    "MCG-NJU/videomae-base-short-ssv2",
    "MCG-NJU/videomae-base-short-finetuned-ssv2",
    "sayakpaul/videomae-base-finetuned-ucf101-subset",
    "nateraw/videomae-base-finetuned-ucf101",
    "MCG-NJU/videomae-base-ssv2",
    "zahrav/videomae-base-finetuned-ucf101-subset",
]


pipe = pipeline(
    task="video-classification",
    model=DEFAULT_MODEL,
    top_k=5,
    frame_sampling_rate=FRAME_SAMPLING_RATE,
)


examples = [
    #["https://www.youtube.com/watch?v=huAJ9dC5lmI"],
    ["https://www.youtube.com/watch?v=wvcWt6u5HTg"],
    ["https://www.youtube.com/watch?v=-3kZSi5qjRM"],
    ["https://www.youtube.com/watch?v=-6usjfP8hys"],
    ["https://www.youtube.com/watch?v=BDHub0gBGtc"],
    ["https://www.youtube.com/watch?v=B9ea7YyCP6E"],
    ["https://www.youtube.com/watch?v=BBkpaeJBKmk"],
    ["https://www.youtube.com/watch?v=BBqU8Apee_g"],
    ["https://www.youtube.com/watch?v=B8OdMwVwyXc"],
    ["https://www.youtube.com/watch?v=I7cwq6_4QtM"],
    ["https://www.youtube.com/watch?v=Z0mJDXpNhYA"],
    ["https://www.youtube.com/watch?v=QkQQjFGnZlg"],
    ["https://www.youtube.com/watch?v=IQaoRUQif14"],
]


def get_video_model_names():
    filter = ModelFilter(
        task='video-classification',
        library='transformers',
    )
    api = HfApi()
    video_models = list(
        iter(api.list_models(filter=filter, sort="downloads", direction=-1))
    )
    video_models = [video_model.id for video_model in video_models]
    return video_models


def select_model(model_name):
    global pipe
    pipe = pipeline(
        task="video-classification",
        model=model_name,
        top_k=5,
        frame_sampling_rate=FRAME_SAMPLING_RATE,
    )


def predict(youtube_url_or_file_path):

    if youtube_url_or_file_path.startswith("http"):
        video_path = download_youtube_video(youtube_url_or_file_path)
    else:
        video_path = youtube_url_or_file_path

    # rearrange sampling rate based on video length and model input length
    num_total_frames = get_num_total_frames(video_path)
    num_model_input_frames = pipe.model.config.num_frames
    if num_total_frames < FRAME_SAMPLING_RATE * num_model_input_frames:
        frame_sampling_rate = num_total_frames // num_model_input_frames
    else:
        frame_sampling_rate = FRAME_SAMPLING_RATE

    gif_path = create_gif_from_video_file(
        video_path, frame_sampling_rate=frame_sampling_rate, save_path="video.gif"
    )

    # run inference
    results = pipe(videos=video_path, frame_sampling_rate=frame_sampling_rate)

    os.remove(video_path)

    label_to_score = {result["label"]: result["score"] for result in results}

    return label_to_score, gif_path


app = gr.Blocks()
with app:
    gr.Markdown("# **<p align='center'>Video Classification with 🤗 Transformers</p>**")
    gr.Markdown(
        """
        <p style='text-align: center'>
        Perform video classification with <a href='https://huggingface.co/models?pipeline_tag=video-classification&library=transformers' target='_blank'>HuggingFace Transformers video models</a>.
        <br> For zero-shot classification, you can use the <a href='https://huggingface.co/spaces/fcakyon/zero-shot-video-classification' target='_blank'>zero-shot classification demo</a>.
        </p>
        """
    )
    gr.Markdown(
        """
        <p style='text-align: center'>
        Follow me for more! 
        <br> <a href='https://twitter.com/fcakyon' target='_blank'>twitter</a> | <a href='https://github.com/fcakyon' target='_blank'>github</a> | <a href='https://www.linkedin.com/in/fcakyon/' target='_blank'>linkedin</a> | <a href='https://fcakyon.medium.com/' target='_blank'>medium</a>
        </p>
        """
    )

    with gr.Row():
        with gr.Column():
            model_names_dropdown = gr.Dropdown(
                choices=VALID_VIDEOCLASSIFICATION_MODELS,
                label="Model:",
                show_label=True,
                value=DEFAULT_MODEL,
            )
            model_names_dropdown.change(fn=select_model, inputs=model_names_dropdown)
            with gr.Tab(label="Youtube URL"):
                gr.Markdown("### **Provide a Youtube video URL**")
                youtube_url = gr.Textbox(label="Youtube URL:", show_label=True)
                youtube_url_predict_btn = gr.Button(value="Predict")
            with gr.Tab(label="Local File"):
                gr.Markdown("### **Upload a video file**")
                video_file = gr.Video(label="Video File:", show_label=True)
                local_video_predict_btn = gr.Button(value="Predict")
        
        with gr.Column():
            video_gif = gr.Image(
                label="Input Clip",
                show_label=True,
            )
        with gr.Column():
            predictions = gr.Label(
                label="Predictions:", show_label=True, num_top_classes=5
            )

    gr.Markdown("**Examples:**")
    gr.Examples(
        examples,
        youtube_url,
        [predictions, video_gif],
        fn=predict,
        cache_examples=True,
    )

    youtube_url_predict_btn.click(
        predict, inputs=youtube_url, outputs=[predictions, video_gif]
    )
    local_video_predict_btn.click(
        predict, inputs=video_file, outputs=[predictions, video_gif]
    )
    gr.Markdown(
        """
        \n Demo created by: <a href=\"https://github.com/fcakyon\">fcakyon</a>.
        <br> Powered by <a href='https://huggingface.co/models?pipeline_tag=video-classification&library=transformers' target='_blank'>HuggingFace Transformers video models</a> .
        """
    )

app.launch()
