import sys, torch
sys.path.insert(0, '.')
from models import Det, Cls, CW, CH
d = Det(); d.load_state_dict(torch.load('det.pt', map_location='cpu')); d.eval()
torch.onnx.export(d, torch.zeros(1, 3, 480, 640), 'det.onnx', input_names=['x'], output_names=['y'],
                  dynamic_axes={'x': {2: 'h', 3: 'w'}, 'y': {2: 'h4', 3: 'w4'}}, opset_version=17, dynamo=False)
c = Cls(); c.load_state_dict(torch.load('cls.pt', map_location='cpu')); c.eval()
torch.onnx.export(c, torch.zeros(2, 3, CH, CW), 'cls.onnx', input_names=['x'], output_names=['num', 'col', 'flip'],
                  dynamic_axes={'x': {0: 'n'}, 'num': {0: 'n'}, 'col': {0: 'n'}, 'flip': {0: 'n'}}, opset_version=17, dynamo=False)
print('ok')
