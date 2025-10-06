# train.py


train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False, num_workers=4)


model = get_model(params['model_name'], num_classes=params['num_classes'], pretrained=params['pretrained'])
model = model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'])
criterion = torch.nn.CrossEntropyLoss()


best_val_acc = 0.0
for epoch in range(params['epochs']):
model.train()
running_loss = 0.0
for batch in train_loader:
imgs, labels = batch
imgs = imgs.to(device)
labels = labels.to(device)
optimizer.zero_grad()
outputs = model(imgs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
running_loss += loss.item() * imgs.size(0)


val_acc = evaluate(model, val_loader, device)
print(f"Epoch {epoch+1}/{params['epochs']} loss={running_loss/len(train_ds):.4f} val_acc={val_acc:.4f}")
if val_acc > best_val_acc:
best_val_acc = val_acc
save_checkpoint(model, Path(params['output_dir']) / f"best_{int(time.time())}.pt")


# Save final
save_checkpoint(model, Path(params['output_dir']) / f"final_{int(time.time())}.pt")




def evaluate(model, loader, device):
model.eval()
correct = 0
total = 0
with torch.no_grad():
for imgs, labels in loader:
imgs = imgs.to(device)
labels = labels.to(device)
outputs = model(imgs)
preds = outputs.argmax(dim=1)
correct += (preds == labels).sum().item()
total += labels.size(0)
return correct / total




if __name__ == '__main__':
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='params_cnn.json')
args = parser.parse_args()
params = json.load(open(args.config))
Path(params['output_dir']).mkdir(parents=True, exist_ok=True)
train_loop(params)