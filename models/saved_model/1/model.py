import tensorflow as tf
import triton_python_backend_utils as pb_utils
from tensorflow_serving.config import model_server_config_pb2
from google.protobuf import text_format

from pathlib import Path

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


def warmup_model(model):
    images = tf.random.uniform((100, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32)
    images = tf.cast(images, tf.uint8)
    jpeg_image = tf.image.encode_jpeg(images[0])
    model.predict_images(images)
    model.predict_jpeg(jpeg_image)


class TritonPythonModel:
    def initialize(self, args):
        with open("/data/models.conf", "r") as f:
            config = f.read()

        model_config = text_format.Parse(
            config, model_server_config_pb2.ModelServerConfig()
        )

        models = {}
        for model in model_config.model_config_list.config:
            # glob only directories
            model_dirs = [
                path for path in Path(model.base_path).glob("*") if path.is_dir()
            ]
            model_dir = model_dirs[0]
            print("Loading model", model.name, model_dir)
            models[model.name] = tf.saved_model.load(model_dir)
            print("Running warmup ...")
            warmup_model(models[model.name])

        self.models = models

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        outputs = [
            {"name": "classes", "data_type": "TYPE_INT64", "dims": [-1]},
            {"name": "scores", "data_type": "TYPE_FP32", "dims": [-1]},
            {"name": "boxes", "data_type": "TYPE_FP32", "dims": [-1, 4]},
            {"name": "logits", "data_type": "TYPE_FP32", "dims": [-1, -1]},
            {"name": "embeddings", "data_type": "TYPE_FP32", "dims": [-1, -1]},
        ]

        for output in outputs:
            auto_complete_model_config.add_output(output)

        return auto_complete_model_config

    def parse_request(self, request):
        model_name = pb_utils.get_input_tensor_by_name(request, "model_name")
        model_name = model_name.as_numpy()[0].decode()

        signature = pb_utils.get_input_tensor_by_name(request, "signature")
        signature = signature.as_numpy()[0].decode()

        predictor = self.models[model_name].signatures[signature]

        inputs = {}
        for name, spec in predictor.structured_input_signature[1].items():
            tensor = pb_utils.get_input_tensor_by_name(request, name)
            inputs[name] = tensor.as_numpy()[0]

        return predictor, inputs

    def batch_predict(self, inputs_list):
        predictions = []
        for predictor, inputs in inputs_list:
            outputs = predictor(**inputs)
            predictions.append({k: v.numpy() for k, v in outputs.items()})
        return predictions

    def prepare_response(self, outputs):
        output_tensors = []
        for key, output in outputs.items():
            output = pb_utils.Tensor(key, output)
            output_tensors.append(output)
        return pb_utils.InferenceResponse(output_tensors=output_tensors)

    def execute(self, requests):
        inputs_list = []
        for request in requests:
            predictor, inputs = self.parse_request(request)
            inputs_list.append((predictor, inputs))

        predictions = self.batch_predict(inputs_list)

        responses = []
        for outputs in predictions:
            responses.append(self.prepare_response(outputs))
        return responses
