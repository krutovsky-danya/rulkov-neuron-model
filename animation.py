import imageio
import cv2


def build_gif(gif_name, filenames):
    with imageio.get_writer(gif_name, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)


def build_video(gif_name, filenames):
    img_array = []
    size = (0, 0)
    for filename in filenames:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(gif_name, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
