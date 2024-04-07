from dataclasses import dataclass
import numpy as np
import tritonclient.grpc as triton_server
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import grpc
import tensorflow as tf


@dataclass(frozen=True)
class TritonSavedModelClient:
    model_name: str = "model"
    url: str = "localhost:8001"

    @property
    def client(self) -> triton_server.InferenceServerClient:
        return triton_server.InferenceServerClient(url=self.url, verbose=False)

    def predict(
        self,
        signature: str,
        *,
        images: np.ndarray = None,
        jpeg_image: bytes = None,
        boxes: np.ndarray = None,
    ):

        input_signature = triton_server.InferInput("signature", [1, 1], "BYTES")
        input_signature.set_data_from_numpy(np.array([[signature.encode()]]))

        input_model_name = triton_server.InferInput("model_name", [1, 1], "BYTES")
        input_model_name.set_data_from_numpy(np.array([[self.model_name.encode()]]))

        inputs = [input_signature, input_model_name]

        if jpeg_image is not None:
            input_tensor = triton_server.InferInput("jpeg_image", [1, 1], "BYTES")
            input_tensor.set_data_from_numpy(np.array([[jpeg_image]]))
            inputs.append(input_tensor)

        if images is not None:
            images = np.expand_dims(images, axis=0)
            input_tensor = triton_server.InferInput("images", images.shape, "UINT8")
            input_tensor.set_data_from_numpy(images)
            inputs.append(input_tensor)

        if boxes is not None:
            boxes = np.expand_dims(boxes, axis=0)
            input_tensor = triton_server.InferInput("boxes", boxes.shape, "FP32")
            input_tensor.set_data_from_numpy(boxes.astype(np.float32))
            inputs.append(input_tensor)

        results = self.client.infer(
            model_name="saved_model",
            model_version="1",
            inputs=inputs,
        )

        return results


@dataclass(frozen=True)
class TFServingGRPCClient:
    model_name: str
    url: str = "localhost:8500"

    def predict(
        self, signature: str, *, images: np.ndarray, timeout: int = 60
    ) -> predict_pb2.PredictResponse:

        options = [("grpc.max_receive_message_length", 1 << 30)]
        channel = grpc.insecure_channel(self.url, options=options)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        grpc_request = predict_pb2.PredictRequest()
        grpc_request.model_spec.name = self.model_name
        grpc_request.model_spec.signature_name = signature

        grpc_request.inputs["images"].CopyFrom(tf.make_tensor_proto(images))

        predict_response = stub.Predict(grpc_request, timeout)
        channel.close()
        return predict_response


def thread_imap1(func, iterable: list, num_workers: int | None = None, desc: str = ""):
    with ThreadPoolExecutor(num_workers) as ex:
        results = list(tqdm(ex.map(func, iterable), total=len(iterable), desc=desc))
    return results


def benchmark_client(
    client, images: np.ndarray, num_workers: int = 4, num_samples: int = 500
):

    client.predict("predict_images", images=images)  # warmup

    def predict_fn(_):
        client.predict("predict_images", images=images)

    _ = thread_imap1(
        predict_fn,
        list(range(num_samples)),
        num_workers=num_workers,
        desc=client.model_name,
    )


def benchmark_clients(
    clients, images: np.ndarray, num_workers: int = 4, num_samples: int = 500
):

    for client in clients:
        client.predict("predict_images", images=images)  # warmup

    def predict_fn(_):
        for client in clients:
            client.predict("predict_images", images=images)

    _ = thread_imap1(predict_fn, list(range(num_samples)), num_workers=num_workers)
