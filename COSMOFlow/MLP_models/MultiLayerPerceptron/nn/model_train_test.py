import torch
import numpy as np
import matplotlib.pyplot as plt
from MultiLayerPerceptron.utilities import norm, norm_inputs, unnorm_inputs, unnorm, get_script_path
from sys import stdout
from pathlib import Path


def model_train_test(data, model, device, n_epochs, n_batches, loss_function, optimizer, verbose=False, return_losses=False, update_every=None, n_test_batches=None, save_best=False, scheduler=None):
    
    if n_test_batches is None:
        n_test_batches = n_batches
    
    xtrain, ytrain, xtest, ytest = data
    model.to(device)

    name = model.name
    path = get_script_path()
    norm_type = model.norm_type
    Path(get_script_path()+f'/models/{name}/').mkdir(parents=True, exist_ok=True)
    if norm_type == 'z-score':
        np.save(path+'/models/'+name+'/xdata_inputs.npy',np.array([xtrain.mean(axis=0), xtrain.std(axis=0)]))
        np.save(path+'/models/'+name+'/ydata_inputs.npy',np.array([ytrain.mean(), ytrain.std()]))
    elif norm_type == 'uniform':
        np.save(path+'/models/'+name+'/xdata_inputs.npy',np.array([np.min(xtrain,axis=0), np.max(xtrain,axis=0)]))
        np.save(path+'/models/'+name+'/ydata_inputs.npy',np.array([np.min(ytrain), np.max(ytrain)]))
    xtest = torch.from_numpy(norm_inputs(xtest, ref_dataframe=xtrain, norm_type=norm_type)).to(device).float()
    ytest = torch.from_numpy(norm(ytest, ref_dataframe=ytrain, norm_type=norm_type)).to(device).float()
    xtrain = torch.from_numpy(norm_inputs(xtrain, ref_dataframe=xtrain, norm_type=norm_type)).to(device).float()
    ytrain = torch.from_numpy(norm(ytrain, ref_dataframe=ytrain, norm_type=norm_type)).to(device).float()

    ytrainsize = len(ytrain)
    ytestsize = len(ytest)

    train_losses = []
    test_losses = []
    rate = []
    # Run the training loop

    datasets = {"train": [xtrain, ytrain], "test": [xtest, ytest]}

    cutoff_LR = n_epochs - 50
    lowest_loss = 1e5
    for epoch in range(n_epochs):  # 5 epochs at maximum
        # Print epoch
        for phase in ['train','test']:
            if phase == 'train':
                model.train(True)
                shuffled_inds = torch.randperm(ytrainsize)

                # Set current loss value
                current_loss = 0.0

                # Iterate over the DataLoader for training data
                # Get and prepare inputs
                inputs, targets = datasets[phase]
                inputs = inputs[shuffled_inds]
                targets = targets[shuffled_inds]

                #targets = targets.reshape((targets.shape[0], 1))

                for i in range(n_batches):
                    for param in model.parameters():
                        param.grad = None
                    outputs = model(inputs[i * ytrainsize // n_batches:(i+1)*ytrainsize // n_batches])
                    loss = loss_function(outputs, targets[i * ytrainsize // n_batches: (i+1)*ytrainsize // n_batches])
                    loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    current_loss += loss.item()

                train_losses.append(current_loss / n_batches)

            else:
                with torch.no_grad():
                    model.train(False)
                    shuffled_inds = torch.randperm(ytestsize)
                    current_loss = 0.0
                    inputs, targets = datasets[phase]
                    inputs = inputs[shuffled_inds]
                    targets = targets[shuffled_inds]

#                     targets = targets.reshape((targets.shape[0], 1))

                    for i in range(n_test_batches):
                        outputs = model(inputs[i * ytestsize // n_batches: (i+1)*ytestsize // n_batches])
                        loss = loss_function(outputs, targets[i * ytestsize // n_batches: (i+1)*ytestsize // n_batches])
                        current_loss += loss.item()

                    test_losses.append(current_loss / n_test_batches)
        if test_losses[-1] < lowest_loss:
            lowest_loss = test_losses[-1]
            if save_best:
                torch.save(model.state_dict(),path+'/models/'+name+'/model.pth')
                
#         if epoch >= cutoff_LR:
#             scheduler.step()
#             rate.append(scheduler.get_last_lr()[0])
#         else:
#             rate.append(learning_rate)
        if verbose:
            stdout.write(f'\rEpoch: {epoch} | Train loss: {train_losses[-1]:.3e} | Test loss: {test_losses[-1]:.3e} ')
        if update_every is not None:
            if epoch % update_every == 0:
                epochs = np.arange(epoch+1)
                plt.semilogy(epochs, train_losses, label='Train')
                plt.semilogy(epochs, test_losses, label='Test', alpha = 0.5)
                plt.ylim([0.01, 1.00])
                plt.legend()
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('Train and Test Loss Across Train Epochs')
                plt.savefig(path+'/models/'+name+'/losses.png')
                #plt.show()
                plt.close()
                
                if not save_best:
                    torch.save(model.state_dict(),path+'/models/'+name+'/model.pth')

        
    if verbose:
        print('\nTraining complete - saving.')
    
    if not save_best:
        torch.save(model.state_dict(),path+'/models/'+name+'/model.pth')

    epochs = np.arange(n_epochs)
    plt.semilogy(epochs, train_losses, label='Train')
    plt.semilogy(epochs, test_losses, label='Test', alpha = 0.5)
    plt.ylim([0.01, 1.00])
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.title('Train and Test Loss Across Train Epochs')
    plt.savefig(path+'/models/'+name+'/losses.png')
    #plt.show()
    plt.close()

    out = (model,)
    if return_losses:
        out += (train_losses, test_losses,)
    return out
