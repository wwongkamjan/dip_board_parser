import cv2
from PIL import Image
import numpy as np
import json
board_local_name = 'board_4'
main_path = '/content/drive/Shareddrives/ALLAN-Diplomacy/diplo_strat youtube'
for i in range (1,50):
  # file_name = f'reddit{i}'
  # try:
  #   with open(f'reddit_posts/ocr_texts/{file_name}.json') as f:
  #       json_string = f.read()
  #       mapped_territories = json.loads(json_string)
  # except:
  #   continue

  file_name = f'frame_{i}'
  try:
    with open(f'{main_path}/ocr/{board_local_name}/{file_name}.json') as f:
        json_string = f.read()
        mapped_territories = json.loads(json_string)
  except:
    continue

  # Color code for different territories
  color_code = {'water': (205, 207, 240),
      'ground': (222, 219, 200),
      'austria': (218, 155, 140),
      'england': (150, 149, 184),
      'france': (196, 195, 209),
      'germany': (157,155,140),
      'italy': (153, 203, 136),
      'russia': (213,155,196),
      'turkey': (210,209,138)}
  color_code = {key: (val[2], val[1], val[0]) for key, val in color_code.items()}
  # Load the image
  image_path = f'{main_path}/{board_local_name}/{file_name}.jpg'  # Update with the actual path

  image = cv2.imread(image_path)
  if image is None:
    # image_path = f'reddit_posts/{file_name}.jpg'
    # image = cv2.imread(image_path)
    continue

  # Function to sample color in the territory bounding box and ignore black (text) areas
  def get_dominant_territory_color(image, bbox, ignore_black_threshold=120):
      x, y, w, h = map(int, bbox)  # Bounding box coordinates and dimensions
      region = image[y:y+h, x:x+w]  # Extract region of the bounding box

      # Mask out dark colors (likely text areas) by filtering pixels with low RGB values
      non_black_pixels = region[(region[:, :, 0] > ignore_black_threshold) |
                                (region[:, :, 1] > ignore_black_threshold) |
                                (region[:, :, 2] > ignore_black_threshold)]

      if len(non_black_pixels) == 0:
          return None  # No valid color found if only black or very dark pixels

      # Calculate the mean color of the remaining pixels
      avg_color = np.mean(non_black_pixels, axis=0)
      return tuple(map(int, avg_color))  # Return as (B, G, R)

  # Function to find the closest color from color_code
  def find_closest_color(sample_color, color_code):
      min_distance = float('inf')
      closest_color_name = None
      for name, color in color_code.items():
          # Calculate Euclidean distance in RGB space
          distance = np.sqrt(sum((sample_color[i] - color[i]) ** 2 for i in range(3)))
          if distance < min_distance:
              min_distance = distance
              closest_color_name = name
      return closest_color_name

  # Dictionary to store the mapping of territory names to their closest color
  territory_color_mapping = {}

  # Process each territory to find the closest color and draw bounding box contours
  for territory_name, bbox in mapped_territories.items():
      sampled_color = get_dominant_territory_color(image, bbox)

      if sampled_color:
          closest_color = find_closest_color(sampled_color, color_code)
          territory_color_mapping[territory_name] = closest_color.upper()

          # Draw the bounding box for the territory on the image
          x, y, w, h = map(int, bbox)
          cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow bounding box for visibility
          cv2.putText(image, closest_color, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Label in red

  # Save the image with bounding box contours to a file
  # output_image_path = f"{file_name}_with_bounding_boxes.png"
  # cv2.imwrite(output_image_path, image)
  # print(f"Contours and labels image saved to {output_image_path}")

  # Save territory color mapping to a JSON file
  output_json_path = f"{main_path}/ocr/{board_local_name}/ter/ter_mapping_{file_name}.json"
  with open(output_json_path, 'w') as json_file:
      json.dump(territory_color_mapping, json_file, indent=4)
