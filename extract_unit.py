import cv2
import numpy as np
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

    # Load the image
    image_path = f'{main_path}/{board_local_name}/{file_name}.jpg'  # Update with the actual path

    image = cv2.imread(image_path)
    if image is None:
      # image_path = f'reddit_posts/{file_name}.jpg'
      # image = cv2.imread(image_path)
      continue

    image_with_contours = image.copy()

    # Unit color codes in BGR format
    unit_color_code = {
    'austria': (201, 20, 10),
    'england': (57, 57, 176),
    'france': (153, 156, 252),
    #  'germany': (89,88,79),
    'italy': (0,170,0),
    'russia': (197,62,190),
    'turkey': (197,196,56)}

    # Convert unit colors from RGB to BGR format
    unit_color_code_bgr = {key: (val[2], val[1], val[0]) for key, val in unit_color_code.items()}

    # Dictionary of text box coordinates, with keys as territory IDs


    # Helper function to find the centroid of a contour
    def get_contour_centroid(contour):
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return None  # Avoid division by zero
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)

    # Function to find the closest territory to a given unit's centroid
    def find_closest_territory(unit_centroid, territory_coords):
        min_distance = float('inf')
        closest_territory = None
        for territory_name, bbox in territory_coords.items():
            # Calculate the center of each territory bounding box
            territory_center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
            distance = np.sqrt((unit_centroid[0] - territory_center[0]) ** 2 + (unit_centroid[1] - territory_center[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_territory = territory_name
        return closest_territory

    # Dictionary to store the mapping of each unit to the closest territory
    unit_to_territory_mapping = {}

    # Process each unit color to find its closest territory
    for unit_name, unit_color in unit_color_code_bgr.items():
        # Create a mask for the unit color with slightly increased tolerance
        lower_bound = np.array([max(0, c - 30) for c in unit_color], dtype=np.uint8)
        upper_bound = np.array([min(255, c + 30) for c in unit_color], dtype=np.uint8)
        mask = cv2.inRange(image, lower_bound, upper_bound)

        # Find contours of the masked areas
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"No contours detected for unit '{unit_name}' color {unit_color}. Adjust tolerance or check image.")

        for contour in contours:
            # Draw each contour on the image copy
            cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 255), 2)  # Draw in yellow for visibility

            # Find the centroid of the contour (unit position)
            centroid = get_contour_centroid(contour)
            if centroid is None:
                continue

            # Draw the centroid as a small circle
            cv2.circle(image_with_contours, centroid, 4, (0, 0, 255), -1)  # Red dot for the centroid

            # Find the closest territory to this unit
            closest_territory = find_closest_territory(centroid, mapped_territories)
            if closest_territory:
                unit_to_territory_mapping[closest_territory] = unit_name.upper()
                # print(f"Unit '{unit_name}' at {centroid} is closest to territory '{closest_territory}'")
                text = unit_name.capitalize()
                font = cv2.FONT_HERSHEY_SIMPLEX
                # Add label for the unit power name near the centroid
                label_position = (centroid[0] + 5, centroid[1] - 5)  # Offset slightly from the centroid
                # Draw "bold" text by overlaying it multiple times for thickness
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:  # Slightly offset in multiple directions
                    cv2.putText(image_with_contours, text, (label_position[0] + dx, label_position[1] + dy),
                                font, 0.5, (255, 0, 0), 1)

    # Save the image with contours and centroids to a file
    output_image_path = f"{file_name}_with_contours.png"
    cv2.imwrite(output_image_path, image_with_contours)
    # print(f"Contours and centroids image saved to {output_image_path}")
    output_json_path = f"{main_path}/ocr/{board_local_name}/unit/unit_mapping_{file_name}.json"
    with open(output_json_path, 'w') as json_file:
        json.dump(unit_to_territory_mapping, json_file, indent=4)
