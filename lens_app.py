from flask import Flask
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import ground_based_dataset as gbd
import resnet_ssl_model as rsm
from data_transforms import Log10, Clamp, AugmentTranslate, WhitenInput
from run_loop import SNTGRunLoop

torch.manual_seed(770715)
torch.cuda.manual_seed_all(770715)
data_path = '/home/milesx/datasets/deeplens'
test_composed = transforms.Compose([WhitenInput(), Clamp(1e-9, 100)])
cpu_data = gbd.GroundBasedDataset(
    data_path, length=1, transform=test_composed, use_cuda=False)
normal_net = rsm.SNTGModel(4)
stng_net = rsm.SNTGModel(4)
normal_model = '/home/milesx/datasets/deeplens/saved_model/ground_based2019-01-21-19-45.pth'
stng_model = '/home/milesx/datasets/deeplens/saved_model/ground_based2019-01-23-13-47.pth'
normal_net.load_state_dict(torch.load(normal_model))
stng_net.load_state_dict(torch.load(stng_net))

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World! Developer!' + str(cpu_data.length)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
