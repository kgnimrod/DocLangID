import cv2
import glob
import os
import argparse

parser = argparse.ArgumentParser(description='PyTorch Lightning ResNet Training')
parser.add_argument('--dataset', default='impact', type=str, dest="dataset",help='dataset to run preprocessing')

def preprocessing_image(image, shapes):
    # Convert to Grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian Adaptive Thresholding
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 15)

    # Gaussian Filter Blurring
    #image = cv2.GaussianBlur(image, (5, 5), 0)

    # Image Rescaling and Resizing
    #image = cv2.convertScaleAbs(image, alpha=255.0 / image.max())
    #image = cv2.resize(image, shapes, fx=0.5, fy=0.5)

    return image

def run_preprocessing():

    args = parser.parse_args()
    if args.dataset == "impact":
        # run the preprocessing for all languages
        languages = ['Dutch', 'Bulgarian', 'Czech', 'Polish', 'Spanish', 'Slovenian']
    else:
        languages = ['en', 'de', 'fr', 'nl']

    for index, language in enumerate(languages):

        os.makedirs(f"/preprocessed/wpi/sales_cat/preprocessed_no_resize_no_gblur/{language}", exist_ok=True)
        os.makedirs(f"/preprocessed/wpi/sales_cat/preprocessed_no_resize_no_gblur/{language}/1/", exist_ok=True)

        print(f"READING IMAGES")
        if args.dataset == "impact":
            paths = glob.glob(f"/dataset/{language}/*.tif")[:500]
        else:
            paths = glob.glob(f"/wpi_raw/{language}/*.jpg")

        print(f"{language} Paths loaded successfully!")
        
        # apply transformations on each element
        for i in range(len(paths)):
            print(f'{i+index*len(paths)}/{len(languages)*len(paths)}')
            image = cv2.imread(paths[i], cv2.IMREAD_UNCHANGED)

            # Convert to Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Gaussian Adaptive Thresholding
            image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 15)

            # Gaussian Filter Blurring
            #image = cv2.GaussianBlur(image, (5, 5), 0)

            # Image Rescaling and Resizing
            #image = cv2.convertScaleAbs(image, alpha=255.0 / image.max())
            #image = cv2.resize(image, (1000, 1000), fx=0.5, fy=0.5)
            # Fast Means Denoising
            #image = cv2.fastNlMeansDenoising(image, image, 15, 10, 50)

            # Save the processed image as .jpg
            # get filename
            filename = os.path.basename(paths[i]).split('.')[0]
            cv2.imwrite(f'/preprocessed/wpi/sales_cat/preprocessed_no_resize_no_gblur/{language}/1/{filename}.png', image)
        print(f"{language} dataset preprocessing finished!")

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    run_preprocessing()