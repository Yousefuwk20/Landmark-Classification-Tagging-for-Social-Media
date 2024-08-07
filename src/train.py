import tempfile
import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from src.helpers import after_subplot

def train_one_epoch(train_dataloader, model, optimizer, loss):
    """
    Performs one train_one_epoch epoch
    """

    if torch.cuda.is_available():
        model.cuda()

    model.train()

    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        # move data to GPU
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        output = model(data)

        loss_value = loss(output, target)

        loss_value.backward()

        optimizer.step()

        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        )

    return train_loss


def valid_one_epoch(valid_dataloader, model, loss):
    """
    Validate at the end of one epoch
    """

    with torch.no_grad():

        model.eval()

        if torch.cuda.is_available():
            model.cuda()

        valid_loss = 0.0
        correct = 0.
        total = 0.

        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            output = model(data)

            loss_value = loss(output, target)

            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
            )
            
            preds = output.data.max(1, keepdim=True)[1]
            correct += torch.sum(torch.squeeze(preds.eq(target.data.view_as(preds))).cpu())
            total += data.size(0)

        valid_accuracy = 100. * correct / total

    return valid_loss, valid_accuracy


def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False):
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    valid_loss_min, _ = valid_one_epoch(data_loaders["valid"], model, loss)
    valid_loss_min_count = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

    for epoch in range(1, n_epochs + 1):
        logs = {}
        
        train_loss = train_one_epoch(data_loaders["train"], model, optimizer, loss)

        valid_loss, valid_accuracy = valid_one_epoch(data_loaders["valid"], model, loss)

        scheduler.step(valid_loss)

        if (valid_loss_min - valid_loss) / valid_loss_min > 0.01:
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")
            if interactive_tracking and (valid_loss_min_count % 3 == 0):
                _, logs["Test Accuracy"] = one_epoch_test(data_loaders['test'], model, loss)
                
            torch.save(model.state_dict(), save_path)

            valid_loss_min = valid_loss

        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["val_acc"] = valid_accuracy
            logs["lr"] = optimizer.param_groups[0]["lr"]

            liveloss.update(logs)
            liveloss.send()

        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.2f}%".format(
                epoch, train_loss, valid_loss, valid_accuracy
            )
        )


def one_epoch_test(test_dataloader, model, loss):
    test_loss = 0.
    correct = 0.
    total = 0.

    with torch.no_grad():
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        for batch_idx, (data, target) in tqdm(
                enumerate(test_dataloader),
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80
        ):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            logits = model(data)

            loss_value = loss(logits, target)

            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - test_loss))

            pred = logits.data.max(1, keepdim=True)[1]

            correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu())
            total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    test_accuracy = 100. * correct / total
    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        test_accuracy, correct, total))

    return test_loss, test_accuracy


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=50, limit=200, valid_size=0.5, num_workers=0)


@pytest.fixture(scope="session")
def optim_objects():
    from src.optimization import get_optimizer, get_loss
    from src.model import MyModel

    model = MyModel(50)

    return model, get_loss(), get_optimizer(model)


def test_train_one_epoch(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
    for _ in range(2):
        lt = train_one_epoch(data_loaders['train'], model, optimizer, loss)
        assert not np.isnan(lt), "Training loss is nan"


def test_valid_one_epoch(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
    for _ in range(2):
        lv, _ = valid_one_epoch(data_loaders["valid"], model, loss)
        assert not np.isnan(lv), "Validation loss is nan"


def test_optimize(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
    with tempfile.TemporaryDirectory() as temp_dir:
        optimize(data_loaders, model, optimizer, loss, 2, f"{temp_dir}/hey.pt")


def test_one_epoch_test(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
    tv, _ = one_epoch_test(data_loaders["test"], model, loss)
    assert not np.isnan(tv), "Test loss is nan"
