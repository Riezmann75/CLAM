import pytorch_lightning
import torchvision
import torch


MODEL_PATH = "pretrained_encoders/tenpercent_resnet18.ckpt"


def load_model_weights(model, weights):

    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print("No weight could be loaded..")
    model_dict.update(weights)
    
    model.load_state_dict(model_dict)

    return model


def load_model(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.resnet18(weights=None)
    torch.serialization.add_safe_globals([pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint])
    torch.serialization.safe_globals([pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint])

    state = torch.load(path, weights_only=True)

    state_dict = state["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "").replace("resnet.", "")] = state_dict.pop(
            key
        )

    model = load_model_weights(model, state_dict)

    model.fc = torch.nn.Sequential()

    model = model.to(device)

    return model


def extract_features(images, device=None):
    # x: (batch_size, 3, 224, 224) shape
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(MODEL_PATH, device=device)
    model.eval()
    with torch.no_grad():
        features = model(images.to(device))  # shape (batch_size, 512)

    return features
