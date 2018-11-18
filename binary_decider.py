import random

image_percentage_chance = 0.7


def decide(img_assesment, txt_assesment):
    if img_assesment == True and txt_assesment == True:
        return True
    if img_assesment == False and txt_assesment == False:
        return False
    if random.random() < image_percentage_chance:
        return img_assesment
    else:
        return txt_assesment
