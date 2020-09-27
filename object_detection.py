import imageio
import torch

from data import BaseTransform
from ssd import build_ssd

"""
The already trained model 'ssd300_mAP_77.43_v2.pth' we are using in here is trained to detect 30-40 objects
Reference: https://github.com/amdegroot/ssd.pytorch
"""


def load_pre_trained_model():
    # loading pre trained model
    # Loads an object saved with :func:`torch.save` from a file. this model is already trained to detect 30-40 objects
    # this will open a tensor that will contain our pre trained model's weights
    pre_trained_model_tensors = torch.load('ssd300_mAP_77.43_v2.pth', map_location=lambda storage, loc: storage)  # loc: storage means return type is storage
    ssd_neural_network = build_ssd(phase='test')  # phase-> train, test (since we will only predict from pre trained model)
    ssd_neural_network.load_state_dict(pre_trained_model_tensors)  # this will attribute pre traied weiights to our ssd neural network
    return ssd_neural_network


def transform_frame(ssd_neural_network):
    transform = BaseTransform(size=ssd_neural_network.size, mean=(104 / 256, 117 / 256, 123 / 256))
    return transform


def detect(frame, transform_frame_function, ssd_neural_network):
    transformed_frame_ndarray = transform_frame_function(image=frame)[0]  # this will transform array into our ssd neural network acceptable size in this case 300x300, first return element is the frame with the right format
    # converting a numpy array to torch tensor
    torch_tensor = torch.from_numpy(transformed_frame_ndarray).permute(2, 0, 1)  # permute(G, R, B) -> converting RGB tp GRB
    # cince neural network do not accept single input like single vector, it accepts only batches
    # we use unsqueeze function to create fake dimention of the batch
    torch_tensor_with_fake_dimention = torch_tensor.unsqueeze(dim=0)  # add at the starting

    with torch.no_grad():
        y = ssd_neural_network(torch_tensor_with_fake_dimention)

    detections = y.data
    frame_height, frame_width = frame.shape[:2]
    scale = torch.Tensor([frame_width, frame_height, frame_width, frame_height])

    return frame


def main():
    # read original video
    reader = imageio.get_reader('demo1.mp4')
    original_video_meta_data = reader.get_meta_data()
    original_video_fps = original_video_meta_data['fps']

    # make writer for an emoty video
    writer = imageio.get_writer('demo1_output.mp4', fps=original_video_fps)
    ssd_neural_network = load_pre_trained_model()
    transform_frame_function = transform_frame(ssd_neural_network)

    i = 1
    for frame in reader:
        predicted_frame = detect(frame=frame, transform_frame_function=transform_frame_function, ssd_neural_network=ssd_neural_network)
        writer.append_data(predicted_frame)
        print(f"frame# {i} is predicted and appended to output video")
        i += 1

    reader.close()
    writer.close()


if __name__ == '__main__':
    main()
