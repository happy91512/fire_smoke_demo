from train import *
import matplotlib.pyplot as plt
model_best = EfficientNet.from_name('efficientnet-b3')
model_best._fc = nn.Linear(in_features=model_best._fc.in_features, out_features=4, bias=True)
model_best.load_state_dict(torch.load('src/detect/model/best.ckpt'))



val_x, val_y = read_file(os.path.join(dataset_dir, 'val'), txt_dir, True)
print(f'validation data size : {len(val_x)}')
val_set = MyDataset(x = val_x, y = val_y, transform = test_transform)
val_loader = DataLoader(dataset = val_set, batch_size=8, shuffle=False)
loss = nn.CrossEntropyLoss()

show_acc = []
show_loss = []
val_acc = []
val_loss = []  
model_best.eval()
for epoch in range(config['num_epochs']):
    epoch_start_time = time.time()
#---------------------validation---------------------
    model_best.eval()
    with torch.no_grad():
        loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
        print(loop)
        for i, (x, y) in loop:  
            x, y = x.to(config['device']), y.to(config['device'])
            model_best = model_best.cuda()
            val_pred = model_best(x)
            y = torch.argmax(y, dim=1)
            batch_loss = loss(val_pred, y)
            acc = (val_pred.argmax(dim=-1) == y.to(device)).float().mean()
            val_loss.append(float(batch_loss))  
            val_acc.append(float(acc))
            loop.set_description(f"Epoch[{epoch}/{config['num_epochs']}]") 
            loop.set_postfix(valiation_loss = batch_loss.item(), valiation_acc = round(float(sum(val_acc)/len(val_acc)), 3))
    show_loss.append(float(sum(val_loss) / len(val_loss)))
    show_acc.append(float(sum(val_acc) / len(val_acc)))

plt.plot(range(config['num_epochs']), show_acc)
plt.savefig('./acc.jpg')


