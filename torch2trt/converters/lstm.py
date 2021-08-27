from torch2trt.torch2trt import *

# credits to: https://github.com/NVIDIA-AI-IOT/torch2trt/issues/144
@tensorrt_converter('torch.nn.LSTM.forward')
def convert_lstm(ctx):
    module = ctx.method_args[0]
    input_tensor = ctx.method_args[1]
    output = ctx.method_return[0]
    layer_count = module.num_layers
    hidden_size = module.hidden_size
    max_seq_length = input_tensor.shape[1] if module.batch_first else input_tensor.shape[0]
    op = trt.RNNOperation.LSTM
    layer = ctx.network.add_rnn_v2(input_tensor._trt, layer_count, hidden_size, max_seq_length, op)
    if module.bidirectional is True:
        layer.direction = trt.RNNDirection.BIDIRECTION
    for i in range(layer_count):
        iw = getattr(module, "weight_ih_l%s" % i).detach().cpu().numpy()
        hw = getattr(module, "weight_hh_l%s" % i).detach().cpu().numpy()
        
        rela_index = 2*i if module.bidirectional is True else i

        layer.set_weights_for_gate(rela_index, trt.RNNGateType.INPUT, True, iw[:hidden_size,:].copy())
        layer.set_weights_for_gate(rela_index, trt.RNNGateType.FORGET, True, iw[hidden_size:hidden_size * 2,:].copy())
        layer.set_weights_for_gate(rela_index, trt.RNNGateType.CELL, True, iw[hidden_size * 2: hidden_size * 3,:].copy())
        layer.set_weights_for_gate(rela_index, trt.RNNGateType.OUTPUT, True, iw[hidden_size * 3:hidden_size * 4,:].copy())

        layer.set_weights_for_gate(rela_index, trt.RNNGateType.INPUT, False, hw[:hidden_size,:].copy())
        layer.set_weights_for_gate(rela_index, trt.RNNGateType.FORGET, False, hw[hidden_size:hidden_size * 2,:].copy())
        layer.set_weights_for_gate(rela_index, trt.RNNGateType.CELL, False, hw[hidden_size * 2: hidden_size * 3,:].copy())
        layer.set_weights_for_gate(rela_index, trt.RNNGateType.OUTPUT, False, hw[hidden_size * 3:hidden_size * 4,:].copy())

        ib = getattr(module, "bias_ih_l%s" % i).detach().cpu().numpy()
        hb = getattr(module, "bias_hh_l%s" % i).detach().cpu().numpy()
        layer.set_bias_for_gate(rela_index, trt.RNNGateType.INPUT, True, ib[:hidden_size].copy())
        layer.set_bias_for_gate(rela_index, trt.RNNGateType.FORGET, True, ib[hidden_size:hidden_size * 2].copy())
        layer.set_bias_for_gate(rela_index, trt.RNNGateType.CELL, True, ib[hidden_size * 2: hidden_size * 3].copy())
        layer.set_bias_for_gate(rela_index, trt.RNNGateType.OUTPUT, True, ib[hidden_size * 3:hidden_size * 4].copy())

        layer.set_bias_for_gate(rela_index, trt.RNNGateType.INPUT, False, hb[:hidden_size].copy())
        layer.set_bias_for_gate(rela_index, trt.RNNGateType.FORGET, False, hb[hidden_size:hidden_size * 2].copy())
        layer.set_bias_for_gate(rela_index, trt.RNNGateType.CELL, False, hb[hidden_size * 2: hidden_size * 3].copy())
        layer.set_bias_for_gate(rela_index, trt.RNNGateType.OUTPUT, False, hb[hidden_size * 3:hidden_size * 4].copy())

        if module.bidirectional is True:
            # ================reverse=====================
            iw_r = getattr(module, "weight_ih_l%s_reverse" % i).detach().cpu().numpy()
            hw_r = getattr(module, "weight_hh_l%s_reverse" % i).detach().cpu().numpy()
    
            layer.set_weights_for_gate(2*i+1, trt.RNNGateType.INPUT, True, iw_r[:hidden_size,:].copy())
            layer.set_weights_for_gate(2*i+1, trt.RNNGateType.FORGET, True, iw_r[hidden_size:hidden_size * 2,:].copy())
            layer.set_weights_for_gate(2*i+1, trt.RNNGateType.CELL, True, iw_r[hidden_size * 2: hidden_size * 3,:].copy())
            layer.set_weights_for_gate(2*i+1, trt.RNNGateType.OUTPUT, True, iw_r[hidden_size * 3:hidden_size * 4,:].copy())
    
            layer.set_weights_for_gate(2*i+1, trt.RNNGateType.INPUT, False, hw_r[:hidden_size,:].copy())
            layer.set_weights_for_gate(2*i+1, trt.RNNGateType.FORGET, False, hw_r[hidden_size:hidden_size * 2,:].copy())
            layer.set_weights_for_gate(2*i+1, trt.RNNGateType.CELL, False, hw_r[hidden_size * 2: hidden_size * 3,:].copy())
            layer.set_weights_for_gate(2*i+1, trt.RNNGateType.OUTPUT, False, hw_r[hidden_size * 3:hidden_size * 4,:].copy())
    
            ib_r = getattr(module, "bias_ih_l%s_reverse" % i).detach().cpu().numpy()
            hb_r = getattr(module, "bias_hh_l%s_reverse" % i).detach().cpu().numpy()
    
            layer.set_bias_for_gate(2*i+1, trt.RNNGateType.INPUT, True, ib_r[:hidden_size].copy())
            layer.set_bias_for_gate(2*i+1, trt.RNNGateType.FORGET, True, ib_r[hidden_size:hidden_size * 2].copy())
            layer.set_bias_for_gate(2*i+1, trt.RNNGateType.CELL, True, ib_r[hidden_size * 2: hidden_size * 3].copy())
            layer.set_bias_for_gate(2*i+1, trt.RNNGateType.OUTPUT, True, ib_r[hidden_size * 3:hidden_size * 4].copy())
    
            layer.set_bias_for_gate(2*i+1, trt.RNNGateType.INPUT, False, hb_r[:hidden_size].copy())
            layer.set_bias_for_gate(2*i+1, trt.RNNGateType.FORGET, False, hb_r[hidden_size:hidden_size * 2].copy())
            layer.set_bias_for_gate(2*i+1, trt.RNNGateType.CELL, False, hb_r[hidden_size * 2: hidden_size * 3].copy())
            layer.set_bias_for_gate(2*i+1, trt.RNNGateType.OUTPUT, False, hb_r[hidden_size * 3:hidden_size * 4].copy())
    lstm_output = layer.get_output(0)
    output._trt = lstm_output
    return