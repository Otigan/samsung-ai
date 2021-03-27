from PIL import Image
def image_resize(image):
    image.thumbnail((224, 224), Image.ANTIALIAS)
    image_size = image.size
    width = image_size[0]
    height = image_size[1]
    if (width != height):
        bigside = width if width > height else height
        background = Image.new('RGB', (bigside, bigside), (255, 255, 255))
        offset = (int(round(((bigside - width) / 2), 0)), int(round(((bigside - height) / 2), 0)))
        background.paste(image, offset)
        return background
    else: return image