dependencies = ['torch']
import torch
from generator import Generator
import os

model_params = {
    'xi_aca5990': {
        'mel_channel': 80,
    },
}


def melgan(model_name='xi_aca5990', pretrained=True, progress=True):
    params = model_params[model_name]
    model = Generator(params['mel_channel'])

    if pretrained:
        checkpoint = './melgan/chkpt/cath_1550.pt'
        model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model_g'])    # MelGAN_v2
        print("\t " + bcolors.red + os.path.basename(checkpoint) + bcolors.reset)

    model.eval(inference=True)

    return model

class bcolors:
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    reset = '\033[0m' #RESET COLOR

if __name__ == '__main__':
    vocoder = torch.hub.load('seungwonpark/melgan', 'melgan')
    mel = torch.randn(1, 80, 234) # use your own mel-spectrogram here

    print('Input mel-spectrogram shape: {}'.format(mel.shape))

    if torch.cuda.is_available():
        print('Moving data & model to GPU')
        vocoder = vocoder.cuda()
        mel = mel.cuda()

    with torch.no_grad():
        audio = vocoder.inference(mel)

    print('Output audio shape: {}'.format(audio.shape))
