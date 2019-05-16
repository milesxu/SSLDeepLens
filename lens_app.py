from flask import Flask
from flask import request
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
normal_model = 'saved_model/ground_based2019-04-04-16-12.pth'
stng_model = 'saved_model/ground_based2019-04-03-15-12.pth'
normal_net.load_state_dict(torch.load(normal_model))
stng_net.load_state_dict(torch.load(stng_model))


def classify_cpu():
    pass


def classify(type, start, length):
    dataset = gbd.GroundBasedDataset(data_path, offset=start, length=length,
                                     transform=test_composed)
    data_loader = DataLoader(dataset, batch_size=length,
                             shuffle=False, pin_memory=False)
    if type == 'normal':
        lens_net = normal_net
    else:
        lens_net = stng_net
    run_loop = SNTGRunLoop(lens_net, test_loader=data_loader)
    return run_loop.test()


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World! Developer!' + str(cpu_data.length)


@app.route('/classify')
def classify_small_cpu():
    processor = request.args.get('processor')
    model = request.args.get('model')
    length = request.args.get('length')
    start = request.args.get('start')
    # return type(length)
    return classify(model, int(start), int(length))


if __name__ == "__main__":
    app.run(debug=True, port=8080)
