import warnings

warnings.filterwarnings("ignore")
import torch, numpy as np, os
from torch import nn
from transformers import AutoModelForImageClassification, AutoConfig, AutoImageProcessor
import saliency.core as saliency
import gradio as gr
import PIL
from glob import glob

model_choice = 3
model_names = [
    "nvidia/mit-b0",
    "facebook/convnext-base-224",
    "microsoft/resnet-18",
    "microsoft/resnet-50",
    "Falconsai/nsfw_image_detection",
    "microsoft/swin-tiny-patch4-window7-224",
    "google/vit-base-patch16-224",
    "juppy44/plant-identification-2m-vit-b",
    "BinhQuocNguyen/food-recognition-model",
    "LucyintheSky/pose-estimation-front-side-back",
    "google/derm-foundation",
    "timm/csatv2"
]
model_name = model_names[model_choice]
device = "cuda" if torch.cuda.is_available() else "cpu"


class Model(nn.Module):
    def __init__(self, MODEL_NAME=model_name):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            MODEL_NAME, finetuning_task="image-classification"
        )
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        self.class_len = self.config.num_labels
        self.id2label = self.config.id2label
        self.label2id = self.config.label2id

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.to(device)
        x = self.model(x)
        return x.logits


def conv_layer_forward_hook(module, input, output):
    """Method from Examples_pytorch.ipynb for the gradcam library https://github.com/PAIR-code/saliency."""
    global last_conv_layer_outputs
    last_conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = (
        torch.movedim(output, 3, 1).detach().cpu().numpy()
    )


def conv_layer_backward_hook(module, grad_input, grad_output):
    """Method from Examples_pytorch.ipynb for the gradcam library https://github.com/PAIR-code/saliency."""
    global last_conv_layer_outputs
    last_conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = (
        torch.movedim(grad_output[0], 3, 1).detach().cpu().numpy()
    )


auto_transformer, class_to_id, id_to_class, last_conv_layer, last_conv_layer_outputs = (
    None,
    None,
    None,
    None,
    None,
)


def swap_models(name):
    global \
        model, \
        auto_transformer, \
        class_to_id, \
        id_to_class, \
        last_conv_layer, \
        last_conv_layer_outputs
    auto_transformer = AutoImageProcessor.from_pretrained(name)
    model = Model(MODEL_NAME=name)
    model = model.to(device).eval()
    # register the hooks for the last convolution layer for Grad-Cam
    named_modules = dict(model.model.named_modules())
    last_conv_layer_name = None
    for name, module in named_modules.items():
        if isinstance(module, torch.nn.Conv2d):
            last_conv_layer_name = name

    last_conv_layer = named_modules[last_conv_layer_name]
    last_conv_layer_outputs = {}

    last_conv_layer.register_forward_hook(conv_layer_forward_hook)
    last_conv_layer.register_backward_hook(conv_layer_backward_hook)
    class_to_id = {v: k for k, v in model.model.config.id2label.items()}
    id_to_class = {k: v for k, v in model.model.config.id2label.items()}


swap_models(model_name)


def saliency_graph(img1, steps=25):
    if img1 is None:
        # Gracefully handle missing image input (e.g., cached example without data)
        blank = np.zeros((224, 224, 3), dtype=np.float32)
        return blank, {}

    if img1.mode != "RGB":
        img1 = img1.convert("RGB")

    img1 = auto_transformer(img1)
    img1 = np.squeeze(np.array(img1.pixel_values))
    if img1.shape[0] < img1.shape[1]:
        img1 = np.moveaxis(img1, 0, -1)
    img1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))

    class_idx_str = "class_idx_str"

    def gradcam_call(images, call_model_args=None, expected_keys=None):
        if (
            not isinstance(images, np.ndarray)
            and not isinstance(images, torch.Tensor)
            and not isinstance(images, PIL.Image.Image)
        ):
            # return two blank images
            im1 = np.zeros((224, 224, 3))
            im2 = np.zeros((224, 224, 3))
            return im1, im2

        if len(images.shape) == 3:
            images = np.expand_dims(images, 0)
        images = torch.tensor(images, dtype=torch.float32)
        images = images.requires_grad_(True)
        target_class_idx = call_model_args[class_idx_str]
        y_pred = model(images)

        if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
            out = y_pred[:, target_class_idx]
            # move actual color channel to the 1st dimension
            # images = torch.movedim(images, 3, 1)
            grads = torch.autograd.grad(out, images, grad_outputs=torch.ones_like(out))
            grads = grads[0].detach().cpu().numpy()
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: grads}
        else:
            hot = torch.zeros_like(y_pred)
            hot[:, target_class_idx] = 1
            model.zero_grad()
            y_pred.backward(gradient=hot, retain_graph=True)
            return last_conv_layer_outputs

    im = img1.astype(np.float32)
    base = np.zeros(img1.shape)

    pred = model(torch.from_numpy(im))
    class_pred = pred.argmax(dim=1).item()
    call_model_args = {class_idx_str: class_pred}
    gradients = saliency.IntegratedGradients()

    s = gradients.GetSmoothedMask(
        im, gradcam_call, call_model_args, x_steps=steps, x_baseline=base, batch_size=25
    )

    smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(s)

    with torch.no_grad():
        output = model.forward(img1)
        output = torch.nn.functional.softmax(output, dim=1)
        output = output.cpu().numpy()
    top_5 = [
        (id_to_class[int(i)], float(output[0][i])) for i in np.argsort(output)[0][-5:][::-1]
    ]

    return smoothgrad_mask_grayscale, dict(top_5)


# gradio Interface
def gradio_interface(img):
    smoothgrad_mask_grayscale, predictions = saliency_graph(img, steps=20)
    return smoothgrad_mask_grayscale, predictions


with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown("# AI Model Explainability")
    gr.Markdown(
        "This function finds the most critical pixels in an image for predicting a class by looking at the pixels models attend to. The best models will ideally make predictions by highlighting the expected object. Poorly generalizable models will often rely on environmental cues instead and forego looking at the most important pixels. Highlighting the most important pixels helps explain/build trust about whether a given model uses the correct features to make its prediction."
    )
    
    with gr.Row():
        with gr.Column():
            test_image = gr.Image(label="Input Image", type="pil")
            model_select_dropdown = gr.Dropdown(
                model_names, label="Model to test", value=model_name, interactive=True
            )
            input_btn = gr.Button("Classify image", variant="primary")
            
        with gr.Column():
            output = gr.Image(label="Pixels used for classification")
            output2 = gr.Label(label="Top 5 Predictions")

    input_btn.click(gradio_interface, test_image, outputs=[output, output2])
    model_select_dropdown.change(swap_models, inputs=[model_select_dropdown]).then(
        gradio_interface, inputs=[test_image], outputs=[output, output2]
    )
    test_image.change(gradio_interface, inputs=[test_image], outputs=[output, output2])
    examples = gr.Examples(
        examples=[x for x in glob("example_images/*") if x.endswith((".jpg", ".png"))],
        inputs=test_image,
        label="Examples",
        fn=gradio_interface,
        cache_examples=True,
        run_on_click=True,
        postprocess=True,
        preprocess=True,
        outputs=[output, output2],
    )


iface.launch(server_name="0.0.0.0")
