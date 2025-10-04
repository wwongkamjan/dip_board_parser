import sys
import supervision
import torch
from ultralytics import YOLO
from PIL import Image
import os
import importlib
import base64
import matplotlib.pyplot as plt
import io
import json


import utils
importlib.reload(utils)

device = 'cuda:0'
from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model

som_model = get_yolo_model(model_path='/fs/nexus-scratch/wwongkam/OmniParser/weights/icon_detect/best.pt')
som_model.to(device)
print('model to {}'.format(device))

caption_model_processor = get_caption_model_processor(model_name="blip2", model_name_or_path="/fs/nexus-scratch/wwongkam/OmniParser/weights/icon_caption_blip2", device=device)
for local_i in range(0,5):
    local_folder = f'board_{local_i}'
    for i in range (1,50):
        cnt = 0
        fileform = 'jpg'
        image_path = f'/fs/nexus-scratch/wwongkam/OmniParser/imgs/{local_folder}/frame_{i}.jpg'

        draw_bbox_config = {
            'text_scale': 0.8,
            'text_thickness': 2,
            'text_padding': 3,
            'thickness': 3,
        }
        BOX_TRESHOLD = 0.05

        try: 
            image = Image.open(image_path)
        except:
            # image_path = f'/fs/nexus-scratch/wwongkam/OmniParser/imgs/cropped_backstabbr/reddit{i}.jpg'
            # image = Image.open(image_path)
            # fileform = 'jpg'
            continue
        
        image_rgb = image.convert('RGB')

        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.5})
        text, ocr_bbox = ocr_bbox_rslt

        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_path, som_model, BOX_TRESHOLD = BOX_TRESHOLD, output_coord_in_ratio=False, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,use_local_semantics=True, iou_threshold=0.1)
        # print(label_coordinates)

        # Decode the base64 image and open it
        image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))

        # Get the original dimensions of the image
        width, height = image.size

        # Set the figure size to match the image's dimensions in inches
        # By dividing by DPI, we ensure the figure size matches the image's pixel size
        dpi = 100  # DPI setting
        figsize = (width / dpi, height / dpi)

        # Create a figure with the same size as the original image
        plt.figure(figsize=figsize, dpi=dpi)
        plt.axis('off')
        plt.imshow(image)

        # Save the image to a file with the same pixel dimensions as the original
        plt.savefig(f'/fs/nexus-scratch/wwongkam/OmniParser/output/{local_folder}/frame_{i}.{fileform}', dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to free memory

        # List of valid territory codes
        valid_territories = [
            'ADR', 'AEG', 'ALB', 'ANK', 'APU', 'ARM', 'BAL', 'BAR', 'BEL', 'BER', 'BLA', 'BOH', 'BOT', 'BRE', 'BUD', 'BUL',
            'BUR', 'CLY', 'CON', 'DEN', 'EAS', 'EDI', 'ENG', 'FIN', 'GAL', 'GAS', 'GRE', 'HEL', 'HOL', 'ION', 'IRI', 'KIE',
            'LON', 'LVN', 'LVP', 'LYO', 'MAO', 'MAR', 'MOS', 'MUN', 'NAF', 'NAO', 'NAP', 'NWY', 'NTH', 'NWG', 'PAR', 'PIC',
            'PIE', 'POR', 'PRU', 'ROM', 'RUH', 'RUM', 'SER', 'SEV', 'SIL', 'SKA', 'SMY', 'SPA', 'STP', 'SWE', 'SYR', 'TRI',
            'TUN', 'TUS', 'TYR', 'TYS', 'UKR', 'VEN', 'VIE', 'WAL', 'WAR', 'WES', 'YOR'
        ]
        # color_code = {'water': (207, 209, 242),
        #  'ground': (221, 219, 198),
        #  'austria': (211, 155, 194),
        #  'england': (153, 152, 186),
        #  'france': (201, 200, 214),
        #  'germany': (156,154,139),
        #  'italy': (153, 203, 136),
        #  'russia': (213,155,196),
        #  'turkey': (210,209,138)}

        # unit_color_code = {
        #  'austria': (204, 0, 0),
        #  'england': (57, 57, 176),
        #  'france': (153, 153, 255),
        #  'germany': (62,61,55),
        #  'italy': (0,170,0),
        #  'russia': (193,35,190),
        #  'turkey': (189,189,12)}

        # # Filtered list
        filtered_territories = [box for box in parsed_content_list if any(terr in box.upper() for terr in valid_territories)]

        # # Dictionary to store the mapping of territory names to coordinates
        mapped_territories = {}

        for entry in filtered_territories:
            # Extract the ID and territory name from the entry
            parts = entry.split(":")
            text_box_id = parts[0].split(" ")[3]  # Extract ID from "Text Box ID 44"
            territory_name = parts[1].strip().upper()  # Extract territory name, e.g., "Ple"
            
            # Check if the ID exists in the text box coordinates dictionary
            if text_box_id in label_coordinates and territory_name in valid_territories:
                # print(f"Territory '{territory_name}' has coordinates {label_coordinates[text_box_id]}")
                mapped_territories[territory_name] = [round(float(value), 3) for value in label_coordinates[text_box_id]]
            
        # Save the dictionary as a JSON file
        with open(f'/fs/nexus-scratch/wwongkam/OmniParser/output/{local_folder}/frame_{i}.json', 'w') as json_file:
            json.dump(mapped_territories, json_file, indent=4)

        print(f"Data saved to /{local_folder}/frame_{i}.json")