from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.Conv2d.forward', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.nn.Conv3d.forward', enabled=trt_version() >= '7.0')
def convert_Conv_trt7(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]

    if ctx.network.num_inputs==0:
        if ctx.network.has_implicit_batch_dimension:
            input_trt = ctx.network.add_input(
                        name='input_0',
                        shape=tuple(input.shape)[1:],
                        dtype=torch_dtype_to_trt(input.dtype),
                    )
        else:
            input_trt = ctx.network.add_input(
                name='input_0',
                shape=(1,1,64,-1),#tuple(input.shape),#[1:],
                dtype=torch_dtype_to_trt(input.dtype),
            )

    else:
        input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    #actually shouldn't need the following lines, bc it was to fix the original input shape [1,1,224,224] from being broadcasted down to [224,224]
    # but that should be handled in add_inputs shape=tuple(torch_input.shape)[1:]
    #problems is that tensor from add inputs isn't coming here
    # if len(input_trt.shape) < 3:
    #     input_trt = broadcast_trt_tensors(ctx.network, [input_trt], 3)[0]
        
    output = ctx.method_return

    input_dim = input.dim() - 2

    kernel_size = module.kernel_size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * input_dim

    stride = module.stride
    if not isinstance(stride, tuple):
        stride = (stride, ) * input_dim

    padding = module.padding
    if not isinstance(padding, tuple):
        padding = (padding, ) * input_dim

    dilation = module.dilation
    if not isinstance(dilation, tuple):
        dilation = (dilation, ) * input_dim

    kernel = module.weight.detach().cpu().numpy()
    
    bias = None #trt.Weights(torch_dtype_to_trt(module.weight.dtype))
    if module.bias is not None:
        bias = module.bias.detach().cpu().numpy()

    layer = ctx.network.add_convolution_nd(
        input=input_trt,
        num_output_maps=module.out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias)
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.dilation_nd = dilation

    if module.groups is not None:
        layer.num_groups = module.groups

    output._trt = layer.get_output(0)



@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.0')
def test_Conv2d_basic_trt7():
    return torch.nn.Conv2d(10, 5, kernel_size=1, stride=1, padding=0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.0')
def test_Conv2d_stride2_trt7():
    return torch.nn.Conv2d(10, 5, kernel_size=1, stride=2, padding=0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.0')
def test_Conv2d_kernel3_trt7():
    return torch.nn.Conv2d(10, 5, kernel_size=3, stride=2, padding=1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224, 224)], enabled=trt_version() >= '7.0')
def test_Conv2d_dilation2_trt7():
    return torch.nn.Conv2d(10, 5, kernel_size=3, stride=1, padding=1, dilation=2)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.0')
def test_Conv3d_basic_trt7():
    return torch.nn.Conv3d(10, 5, kernel_size=1, stride=1, padding=0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.0')
def test_Conv3d_stride2_trt7():
    return torch.nn.Conv3d(10, 5, kernel_size=1, stride=2, padding=0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.0')
def test_Conv3d_kernel3_trt7():
    return torch.nn.Conv3d(10, 5, kernel_size=3, stride=2, padding=1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 64, 64, 64)], enabled=trt_version() >= '7.0')
def test_Conv3d_dilation2_trt7():
    return torch.nn.Conv3d(10, 5, kernel_size=3, stride=1, padding=1, dilation=2)
