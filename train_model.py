from MODEL import *

target = []
predictions = []


# Функция обучения сети
# def train_model(model, loss, optimizer, scheduler, num_epochs, path_weigh_save):
def train_model(model, train_dataloader, val_dataloader, loss, loss_ls, optimizer, scheduler, num_epochs,
                path_weigh_save, model_name: str):
    train_loss = []
    running_acc_train_1out = 0.
    train_acc_1out = []
    val_loss_1out = []
    val_acc_1out = []
    running_acc_val_1out = 0.
    # model = model.cuda()
    loss_common = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                # scheduler.step()
                model.train()  # Set model to training mode
            else:

                dataloader = val_dataloader
                model.eval()  # Set model to evaluate mode

            running_loss_1out = 0.
            running_acc_1out = 0.



            # Iterate over data.
            for j, (features, target) in enumerate(dataloader):
                features = features.to(torch.float)
                target = target.to(torch.float)
                inputs = features  # .cuda()
                labels = target  # .cuda()
                preds = torch.tensor([])
                preds = preds  # .cuda()
                # print(inputs.shape)

                optimizer.zero_grad()
                preds_1 = model(features)
                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    # with torch.autograd.set_detect_anomaly(True):
                    for i in range(features.shape[0]):
                        preds_1 = model(inputs[i])
                        preds = torch.cat((preds, preds_1.view(-1, 1)), 0)

                    loss_value_1_out = torch.sqrt((preds - labels) ** 2).sum()
                    loss_common += torch.sqrt((preds - labels) ** 2).sum()
                    # loss_value_1_out = loss(preds, labels)
                    # loss_common += loss(preds, labels)

                    if phase == 'train':

                        loss_value_1_out.backward()
                        optimizer.step()

                        train_loss.append(loss_value_1_out.item())

                        running_acc_train_1out += (preds == labels.data).float().mean()
                        train_acc_1out.append(running_acc_train_1out)

                    elif phase == 'val':

                        val_loss_1out.append(loss_value_1_out.item())
                        running_acc_val_1out += (preds == labels.data).float().mean()

                        val_acc_1out.append(running_acc_val_1out)

                # statistics
                # running_loss_2out += loss_value_2_out.item()
                running_loss_1out += loss_value_1_out.item()

                running_acc_1out += (preds_1 == labels.data).float().mean()

            epoch_loss_1out = running_loss_1out / len(dataloader)
            epoch_acc_1out = running_acc_1out / len(dataloader)

            print('{} OUTPUT: Loss: {:.4f} '.format(phase, epoch_loss_1out), flush=True)
            # wandb.log({f"loss": loss_value_1_out, "epoch": epoch, 'common_loss': epoch_loss_1out})
            # torch.save(model.state_dict(), path_weigh_save + str(epoch))
            if epoch % 100 == 0:
                torch.save(model.state_dict(), f"{path_weigh_save}log_model_epoch{epoch}_loss{epoch_loss_1out :.3f}.pt")
    return model, train_loss, val_loss_1out, train_acc_1out, val_acc_1out
