import torch
from src.cir_mbv2 import tiny_nas_mobilenetv2
from src.cir_layer import CirConv2d

if __name__ == '__main__':
    model = tiny_nas_mobilenetv2()
    checkpoint = torch.load('/home/xts/code/njeans/MyViT/pretrained/mbv2_tiny_nas.pth.tar')
    print(checkpoint.keys())
    print(checkpoint['arch'])
    old_state_dict=checkpoint['state_dict']
    new_state_dict = {}
    for old_key, old_value in old_state_dict.items():
        # 如果旧的键在新的模型中存在，那么直接使用旧的值
        if old_key in model.state_dict() and old_value.shape == model.state_dict()[old_key].shape:
            new_state_dict[old_key] = old_value
        else:
            # fill the shape with 0 for old_key
            new_value = torch.zeros_like(model.state_dict()[old_key])
            for i in range(new_value.shape[0]):
                if i<old_value.shape[0]:
                    new_value[i] = old_value[i]
                else:
                    new_value[i] = torch.tensor(-100000)
            print("old key, old value:", old_key, old_value)
            # print("new key, new value:", model.state_dict()[old_key])
            # print("new value:", new_value)
            new_state_dict[old_key] = new_value
        # 如果旧的键在新的模型中不存在，那么需要找到新的键
    checkpoint['state_dict'] = new_state_dict
    torch.save(checkpoint, '/home/xts/code/njeans/MyViT/pretrained/mbv2_tiny_nas2.pth.tar')
    # print(model)