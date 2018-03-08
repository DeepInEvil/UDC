import os


def save_model(model, name):
    if not os.path.exists('models/'):
        os.makedirs('models/')

    torch.save(model.state_dict(), 'models/{}.bin'.format(name))
