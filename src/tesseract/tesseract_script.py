import sys
import time
import pytesseract
from langdetect import detect
import argparse
import json

"""
This is script for using the OCR based approach for language detection.
"""

def run_tesseract(data):
    tesserect_data = pytesseract.image_to_data(
        data[1], lang=data[0], output_type=pytesseract.Output.DICT)
    return (data[0], tesserect_data)

def detect_language(image_path, languages):
    # start time bc important processing is starting
    start_time = time.time()

    # run tesseract for each language
    tesseract_data = []
    for language in languages:
        tesseract_data.append(run_tesseract((language, image_path)))

    # calculate confidences for each language
    results = []
    for (language, extracted_data) in tesseract_data:
        # calculate the average confidence of the extraction
        sum_conf = 0
        numb_conf = 0
        for conf in extracted_data['conf']:
            # this has to be parsed to float first 
            # because of "invalid literal for int() with base 10" error
            if int(float(conf)) == -1:
                continue
            sum_conf += float(conf)
            numb_conf += 1
        results.append(
            (language, sum_conf / numb_conf, extracted_data['text']))

    # sort the contents by confidence
    results.sort(key=lambda x: x[1], reverse=True)

    # create the text and detect the language
    text = ' '.join(results[0][2])

    if args.verbose:
        print(f'Detected text: {text}')

    detected_language = detect(text)

    # stop time bc important processing is over
    end_time = time.time()

    if args.verbose:
        print(f'Confidences by language:')
        for (language, confidence, text) in results:
            print(f'\t{language}: {confidence}')

    if args.verbose:
        print(f'===============================================')
        print(f'Detected language: {detected_language}')
        print(f'===============================================')
        print(f'Processing time: {end_time - start_time} seconds')
        print(f'===============================================')
    else:
        print(f'Detected language: {detected_language}')


# ==============================================
# ==================== MAIN ====================
if __name__ == '__main__':
    # get the arguments
    arg_parser = argparse.ArgumentParser(
        description='OCR based language detection')
    arg_parser.add_argument(
        '-v', '--verbose', action='store_true', help='show more information about the process')
    arg_parser.add_argument('-l', '--languages', nargs='+',
                            required=True, help='languages to use for detection')
    arg_parser.add_argument(
        '-p', '--path', '--image', required=True, help='specify the path to the image')
    args = arg_parser.parse_args()

    # start the script
    if args.verbose:
        print(f'Running the language detection algorithm...')
        print(f'Using the following languages: {args.languages}')

    detect_language(args.path, args.languages)
