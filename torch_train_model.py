from tqdm.notebook import tqdm
import torch

def train_model(net:torch.nn.Module=None,
                dataloaders_dict:dict=None,
                criterion:torch.nn.Module=None,
                optimizer:torch.optim=None, 
                num_epochs:int=2):
    
    """モデルを学習させる関数"""
    losses = {'train': [], 'val': []}
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')

        for phase in ['val', 'train']:
            if phase == 'train':
                net.train()  # モデルを訓練モード
            else:
                net.eval()   # モデルを検証モード

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
            #if (epoch == 0) and (phase == 'train'):
            #    continue

            # データローダーからミニバッチを取り出すループ
            for inputs, labels in tqdm(dataloaders_dict[phase]):
            #for inputs, labels in dataloaders_dict[phase]:
                # optimizerを初期化
                optimizer.zero_grad()
                labels0 = labels[0]

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels0)  # 損失を計算
                    _, preds = torch.max(outputs, 1)    # ラベルを予測
  
                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # イタレーション結果の計算
                    # lossの合計を更新
                    epoch_loss += loss.item() * inputs.size(0)  
                    # 正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels0.data)

            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            print(f'{phase} 損失値:{epoch_loss:.3f} 精度:{epoch_acc:.3f}')
            losses[phase].append(epoch_loss)
    return losses
    