from model import Generator
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

def create_gauss(rows, cols, input_dim=100):
    return torch.randn(rows*cols, input_dim)

def plot(list_img):
    # Plot figure
    plt.figure(figsize=(12, 9))
    for i in range(25):
        img = f"list_img{i}"
        plt.subplot(5, 5, i+1)
        plt.axis(False)
        img = list_img[i]
        plt.imshow(img, cmap="Greys")

    plt.savefig('assets/test.png', bbox_inches='tight')

def main():
    NetG = Generator()
    NetG.load_state_dict(torch.load("model.pth"))
    NetG.eval()

    with torch.inference_mode():
        output_vector = NetG(create_gauss(5, 5, 100))
        output = output_vector * 0.5 + 0.5

    list_img_pil = [T.ToPILImage()(torch.reshape(output[i], (28, 28))) for i in range(output.size(0))]

    plot(list_img_pil)

if __name__ == "__main__":
    main()