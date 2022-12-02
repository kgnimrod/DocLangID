import matplotlib.pyplot as  plt
import torch


def trainer(model, epochs, num_batches, dataset, optimizer, batch_size):
    # Set the model in training mode
    model.train()

    loss_per_10_steps = []
    for epoch in range(1, epochs + 1):
        print('Running epoch: {}'.format(epoch))
        running_loss = 0

        #out = display(progress(1, num_batches + 1), display_id=True)
        for i in range(num_batches):

            # clear out the gradients of all Variables
            optimizer.zero_grad()

            # Forward propogation
            output = model.forward(torch.stack(dataset[i*batch_size:i*batch_size+batch_size]['image']))
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(output, dataset[i*batch_size:i*batch_size+batch_size]['label'])
            running_loss += loss

            if i % 10 == 0: loss_per_10_steps.append(loss)
            #out.update(progress(loss, i, num_batches + 1))

            # calculating the gradients
            loss.backward()

            # updating the params
            optimizer.step()

        running_loss = running_loss / int(num_batches)
        print('Epoch: {} , Running loss: {}'.format(epoch, running_loss))

        # plot the loss
        steps = [i for i in range(len(loss_per_10_steps))]
        plt.plot(steps, loss_per_10_steps)
        plt.title(f'Loss curve for ResNet18 trained for {epochs} epochs')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.show()
    return model