from tqdm.notebook import tqdm
import torch

def train_model(net:torch.nn.Module=None,
                dataloaders_dict:dict=None,
                criterion:torch.nn.Module=None,
                optimizer:torch.optim=None, 
                num_epochs:int=2,
                device:str="cuda" if torch.cuda.is_available() else "cpu"):
    
    """モデルを学習させる関数"""
    losses = {'train': [], 'val': []}
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')

        for phase in ['val', 'train']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            # データローダーからミニバッチを取り出すループ
            for inputs, labels in tqdm(dataloaders_dict[phase]):
            #for inputs, labels in dataloaders_dict[phase]:
                inputs.to(device)
                labels0 = labels[0].to(device)
                optimizer.zero_grad()   # optimizerを初期化

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels0)  # 損失を計算
                    _, preds = torch.max(outputs, 1)    # ラベルを予測
  
                    if phase == 'train': # 訓練時はバックプロパゲーション
                        loss.backward()
                        optimizer.step()

                    
                    epoch_loss += loss.item() * inputs.size(0)          # 損失値合計を更新
                    epoch_corrects += torch.sum(preds == labels0.data)  # 正解数合計を更新

            # エポック毎の損失値と正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            print(f'{phase} 損失値:{epoch_loss:.3f} 精度:{epoch_acc:.3f}')
            losses[phase].append(epoch_loss)
    return losses
    