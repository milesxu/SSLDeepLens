from flask import Flask
import ground_based_dataset as gbd

data_path = '/home/milesx/datasets/deeplens'
cpu_data = gbd.GroundBasedDataset(data_path, length=1, use_cuda=False)


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World! Developer!' + str(cpu_data.length)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
