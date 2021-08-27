"""
Remeber to conda activate easyocr before running this! 
"""

from cv2 import rotatedRectangleIntersection
import easyocr
import torch
import torch.nn as nn
from torch2trt import torch2trt
import time



from torch2trt import *
@tensorrt_converter('torch.Tensor.__hash__')
def convert_hash(ctx):
    input_a = ctx.method_args[0]
    output = ctx.method_return
    output = id(input_a) #i don't think output changes... but ok

@tensorrt_converter('torch.Tensor.get_device') #doesn't exist in pytorch 1.9 i think..
def convert_hash(ctx):
    input_a = ctx.method_args[0]
    output = ctx.method_return
    output = input_a.get_device() #i don't think output changes... but ok

@tensorrt_converter('torch.zeros')
def convert_zeros(ctx):
    # input_a = ctx.method_args[0]
    # output = ctx.method_return
    # output._trt = add_missing_trt_tensors(ctx.network, [output])
    pass




"""
#short example torch model
class VGG_FeatureExtractor(nn.Module):
    def __init__(self, input_channel, output_channel=256):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
             nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))
    def forward(self, input):
        return self.ConvNet(input)
    
model = nn.DataParallel(VGG_FeatureExtractor(1))
model.eval()
#x = torch.ones((1, 128, 224, 224),dtype=torch.float).cuda()
x = torch.ones((1,1,64,320),dtype=torch.float).to('cuda')
model_trt = torch2trt(model, [x])
print('running example torch model',model_trt(x))
"""
def profile(model,dummy_input):
    iters=10
    #note: taken from  volksdep benchmark.py
    with torch.no_grad():
            # warm up
            for _ in range(10):
                model(dummy_input)

            # throughput evaluate
            torch.cuda.current_stream().synchronize()
            t0 = time.time()
            for _ in range(iters):
                model(dummy_input)
            torch.cuda.current_stream().synchronize()
            t1 = time.time()
            throughput = 1.0 * iters / (t1 - t0)

            # latency evaluate
            torch.cuda.current_stream().synchronize()
            t0 = time.time()
            for _ in range(iters):
                model(dummy_input)
                torch.cuda.current_stream().synchronize()
            t1 = time.time()
            latency = round(1000.0 * (t1 - t0) / iters, 2)
    print("throughput: %.3f \t latency: %.3f"% (throughput,latency))


reader = easyocr.Reader(['en'],gpu=True) # need to run only once to load model into memory
print(reader)
print(reader.model_storage_directory)

    # layer = ctx.network.add_convolution_nd(input=input_trt,num_output_maps=module.out_channels,kernel_shape=kernel_size,kernel=kernel,bias=bias)
    #     input_trt2 = ctx.network.add_input(name='input_1',shape=tuple(input.shape)[1:],dtype=torch_dtype_to_trt(input.dtype),)
    # layer2 = ctx.network.add_convolution_nd(input=input_trt,num_output_maps=module.out_channels,kernel_shape=kernel_size,kernel=kernel,bias=bias)
"""
#detector: 
#print(reader.detector)
y = torch.ones((1, 3, 224, 224),dtype=torch.float).cuda()
y2 = torch.randn((1, 3, 480, 928),dtype=torch.float).cuda()
y3 = torch.randn((1, 3, 896, 1312),dtype=torch.float).cuda()

print("Detector:")
#print("Before Conversion:")
#profile(reader.detector, y) #throughput: 12.386 	 latency: 84.190
model_trt_detect = torch2trt(reader.detector, [y])
torch.save(model_trt_detect.state_dict(),'easyocr_detect_JAX.pth')
# model_trt_detect = TRTModule()
# model_trt_detect.load_state_dict(torch.load('easyocr_detect_JAX.pth'))
print("After Conversion")
#profile(model_trt_detect, y) #throughput: 24.737 	 latency: 48.990
out = model_trt_detect(y)
print('running detect inference 1',out[0].shape)
print('\n')
out2 = model_trt_detect(y2)
print('running detect inference 2',out2[0].shape)
print('\n')
out3 = model_trt_detect(y3)
print('running detect inference 3',out3[0].shape)

"""
#recognizer
print("\nRecognizer:")
#print(reader.recognizer)
#x = torch.ones((1, 1, 224, 224),dtype=torch.float).cuda()
x = torch.randn((1,1,64,896),dtype=torch.float).to('cuda')
#recognizer_cuda = reader.recognizer.to('cuda')
#print(recognizer_cuda(x)[0].size)
reader.recognizer.eval()
#print("Before Conversion:")
#profile(reader.recognizer, x) #throughput: 36.912 	 latency: 24.610
# model_trt_rec = torch2trt(reader.recognizer, [x])#, use_onnx=True)
# torch.save(model_trt_rec.state_dict(),'easyocr_recognize_JAX.pth')
# model_trt_rec = TRTModule()
# model_trt_rec.load_state_dict(torch.load('easyocr_recognize_JAX.pth'))

x = torch.ones((1, 1, 64, 224)).cuda()

# convert to TensorRT feeding sample data as input
opt_shape_param = [
    [
        [1, 1, 64, 128],   # min
        [1, 1, 64, 256],   # opt
        [1, 1, 64, 896]    # max
    ]
]
model_trt_rec = torch2trt(reader.recognizer, [x], fp16_mode=False, opt_shape_param=opt_shape_param)
print("After Conversion")
print('running recognize inference')
out_trt = model_trt_rec(x)
out = reader.recognizer(x)
print("Testing Model Difference:")
print(torch.max(torch.abs(out_trt - out)))

#profile(model_trt_rec,x) #throughput: 2296.110 	 latency: 0.450

# print("\nOriginal Network Result:")
# result = reader.readtext('/workdir/cstr-vedastr/images/0_ronaldo_ronaldo.png')
# print(result)
