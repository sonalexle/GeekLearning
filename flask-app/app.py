import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template, url_for, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
import torch
import os


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

models = {
    "svm": torch.jit.load("traced-models/svm.pt"),
    "mlp": torch.jit.load("traced-models/mlp.pt"),
    "resnet": torch.jit.load("traced-models/resnet.pt")
}

model_names = {"svm": "SVM", "mlp": "MLP", "resnet": "ResNet-50"}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def transform_image(infile):
    input_transforms = [transforms.Resize(255),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    my_transforms = transforms.Compose(input_transforms)
    try:
        image = Image.open(infile).convert('RGB')
    except Exception as e:
        return False, e
    timg = my_transforms(image)
    timg.unsqueeze_(0)
    return True, timg


def get_prediction(model, input_tensor):
    outputs = model(input_tensor)
    pred_idx = outputs.argmax(dim=1).item()
    return pred_idx, outputs.squeeze().tolist()


def render_prediction(pred_idx, pred_probs, img_class=["Anime", "Cartoon", "Pokemon"]):
    class_to_prob = {classname: round(prob, 4)*100 for classname, prob in zip(img_class, pred_probs)}
    pred_class = img_class[pred_idx]
    return class_to_prob, pred_class


app = Flask(__name__)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')


@app.route('/', methods=['GET'])
def upload_file():
   return render_template('base.html', model_names=model_names)


@app.route('/', methods=['POST'])
def predict():
    error_msg = ""
    if "model-choice" not in request.form:
        error_msg = " No model choice part."
    if "image_uploads" not in request.files:
        error_msg += "No file part."
    else:
        file = request.files['image_uploads']
        model_choice = request.form['model-choice']
        if file.filename == '':
            error_msg = "Error: you might have not uploaded an image, or the name of the image is empty."
        elif not allowed_file(file.filename):
            error_msg = f"Error: the file did not have one of the following extensions: {(', ').join(ALLOWED_EXTENSIONS)}."   
        elif file and allowed_file(file.filename) and model_choice:
            succeeded, input_tensor = transform_image(file)
            if not succeeded:
                error_msg = f"Error while processing the image: {input_tensor}. Was the file really an image?"
            else:
                pred_idx, pred_probs = get_prediction(models[model_choice], input_tensor)
                class_to_prob, pred_class = render_prediction(pred_idx, pred_probs)
                probs_df = pd.DataFrame \
                            .from_dict(class_to_prob, orient="index", columns=["Predicted probability (%)"]) \
                            .reset_index() \
                            .rename(columns={"index": "Predicted category"})
                df_html = probs_df.to_html(index=False, classes='data', header="true")
                pred_prob = round(class_to_prob[pred_class], 2)
                return render_template(
                    'results.html',
                    tables=[df_html],
                    model_name=model_names[model_choice],
                    class_prob_pair={pred_class: pred_prob}
                )
        else:
            error_msg = "Unknown error."
    return render_template('errors.html', error_msg=error_msg)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
