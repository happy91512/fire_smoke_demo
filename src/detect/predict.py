from train import *
import cv2
model_best = EfficientNet.from_name('efficientnet-b3')
model_best._fc = nn.Linear(in_features=model_best._fc.in_features, out_features=4, bias=True)
model_best.load_state_dict(torch.load('src/detect/model/best.ckpt'))

x = cv2.imread('src/detect/dataset/images/other_(1).jpg')
x = cv2.resize(x, (128, 128))
x = test_transform(x)
x = x.to(device)
x = x.unsqueeze(0)
model_best = model_best.cuda()
pred = model_best(x)
print('class: ',int(pred.argmax(-1).cpu()))