import base64
import io
import os

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template
from PIL import Image


app = Flask(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "vae_decoder.keras"
)

decoder = None
LATENT_DIM = None


def load_decoder():
    global decoder, LATENT_DIM

    if decoder is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        decoder = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        input_shape = decoder.input_shape

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        LATENT_DIM = int(input_shape[-1])

        print("VAE decoder loaded.")
        print("Decoder input shape:", decoder.input_shape)
        print("Decoder output shape:", decoder.output_shape)
        print("Latent dimension:", LATENT_DIM)

    return decoder


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    try:
        model = load_decoder()
        return jsonify({
            "status": "ok",
            "model_loaded": True,
            "latent_dim": LATENT_DIM,
            "decoder_input_shape": [
                str(x) for x in (
                    model.input_shape
                    if isinstance(model.input_shape, tuple)
                    else model.input_shape
                )
            ],
            "decoder_output_shape": [
                str(x) for x in (
                    model.output_shape
                    if isinstance(model.output_shape, tuple)
                    else model.output_shape
                )
            ]
        })
    except Exception as exc:
        return jsonify({
            "status": "error",
            "model_loaded": False,
            "error": str(exc)
        }), 500


@app.route("/generate", methods=["POST"])
def generate():
    try:
        model = load_decoder()

        # Standard VAE prior:
        # z ~ N(0, I)
        latent_vector = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=(1, LATENT_DIM)
        ).astype(np.float32)

        print(
            "Random latent sample:",
            np.round(latent_vector[0], 4).tolist()
        )

        generated_image = model.predict(
            latent_vector,
            verbose=0
        )[0]

        generated_image = np.squeeze(generated_image)

        if generated_image.ndim != 2:
            raise ValueError(
                f"Unexpected decoder output shape: "
                f"{generated_image.shape}"
            )

        generated_image = np.clip(
            generated_image,
            0.0,
            1.0
        )

        generated_image = (
            generated_image * 255
        ).astype(np.uint8)

        image = Image.fromarray(
            generated_image,
            mode="L"
        )

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        image_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return jsonify({
            "success": True,
            "latent": latent_vector[0].tolist(),
            "image": image_base64
        })

    except Exception as exc:
        print("ERROR:", repr(exc))
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500


if __name__ == "__main__":
    app.run(
        debug=False,
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5103"))
    )
