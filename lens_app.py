from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import ground_based_dataset as gbd
import resnet_ssl_model as rsm
from data_transforms import Log10, Clamp, AugmentTranslate, WhitenInput
from run_loop import SNTGRunLoop

torch.manual_seed(770715)
torch.cuda.manual_seed_all(770715)
# data_path = '/home/milesx/datasets/deeplens'
data_path = '/data/mingx/datasets'
test_composed = transforms.Compose([WhitenInput(), Clamp(1e-9, 100)])
cpu_data = gbd.GroundBasedDataset(
    data_path, length=1, transform=test_composed, use_cuda=False)
normal_net = rsm.SNTGModel(4)
sntg_net = rsm.SNTGModel(4)
normal_model = 'saved_model/ground_based2019-04-03-15-12.pth'
sntg_model = 'saved_model/ground_based2019-05-17-12-02.pth'
normal_net.load_state_dict(torch.load(normal_model))
sntg_net.load_state_dict(torch.load(sntg_model))


def classify_cpu():
    pass


def classify(processor, type, start, length):
    # print(processor, type, start, length)
    if type == 'normal':
        lens_net = normal_net
    else:
        lens_net = sntg_net
    if processor == 'gpu':
        # print('cuda used!')
        dataset = gbd.GroundBasedDataset(data_path, offset=start, length=length,
                                         transform=test_composed)
        data_loader = DataLoader(dataset, batch_size=length,
                                 shuffle=False, pin_memory=False)
        run_loop = SNTGRunLoop(lens_net, test_loader=data_loader)
    else:
        dataset = gbd.GroundBasedDataset(data_path, offset=start, length=length,
                                         transform=test_composed, use_cuda=False)
        data_loader = DataLoader(dataset, batch_size=length,
                                 shuffle=False, pin_memory=True)
        run_loop = SNTGRunLoop(
            lens_net, test_loader=data_loader, has_cuda=False)

    # run_loop = SNTGRunLoop(stng_net, test_loader=data_loader)
    result, label, time, accuracy = run_loop.test()
    return result, label, time, accuracy


app = Flask(__name__)
CORS(app)


@app.route('/')
def hello_world():
    return 'Hello, World! Developer!' + str(cpu_data.length)


@app.route('/classify')
def classify_api():
    processor = request.args.get('processor').strip()
    model = request.args.get('model').strip()
    length = request.args.get('length')
    start = request.args.get('start')
    # return type(length)
    result, label, time, accuracy = classify(
        processor, model, int(start), int(length))
    return jsonify({
        'result': result, 'label': label, 'time': time, 'accuracy': accuracy})


if __name__ == "__main__":
    app.run(debug=True, port=8080)
