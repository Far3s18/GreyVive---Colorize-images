import gradio as gr
from model.predict import predict_image

def launch_interface():
    with gr.Blocks(title="Colorization App") as demo:
        gr.Markdown(
            """
            # 🎨 Colorize Your Grayscale Images
            Upload a grayscale photo and see it colorized using a custom-trained CNN Autoencoder.

            **Model**: Trained on CelebA  
            **Architecture**: U-Net style autoencoder  
            **Input**: 128x128 grayscale  
            **Output**: Reconstructed RGB prediction
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column():
                input_img = gr.Image(
                    type="filepath",
                    label="Grayscale Image",
                    image_mode='L',
                    height=256,
                    width=256
                )
            with gr.Column():
                output_img = gr.Image(
                    label="Colorized Output",
                    image_mode='RGB',
                    height=256,
                    width=256
                )

        gr.Examples(
            examples=["/home/fares-fadi/Desktop/Fares/Portfolio/greyVive/assets/input.jpg"],
            inputs=input_img,
            label="Example Input"
        )

        submit_btn = gr.Button("✨ Colorize")

        submit_btn.click(fn=predict_image, inputs=input_img, outputs=output_img)

    demo.launch(show_error=True, inbrowser=True)
