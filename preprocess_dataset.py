import pandas as pd
import numpy as np
import math

# Read the original dataset
data = pd.read_csv('metadata.csv', header=None)
data.columns = ['image', 'left_x', 'left_y', 'right_x', 'right_y']

# Calculate average y coordinate and roll angle for each horizon line
processed_data = []

for _, row in data.iterrows():
    image_path = row['image']
    left_x, left_y = row['left_x'], row['left_y']
    right_x, right_y = row['right_x'], row['right_y']
    
    # Calculate average y coordinate
    avg_y = (left_y + right_y) / 2
    
    # Calculate roll angle (in radians, then convert to degrees)
    # tan(angle) = (right_y - left_y) / (right_x - left_x)
    dx = right_x - left_x
    dy = right_y - left_y
    
    # Handle vertical lines
    if abs(dx) < 1e-6:
        roll_angle = 90.0 if dy > 0 else -90.0
    else:
        roll_angle = math.degrees(math.atan2(dy, dx))
    
    processed_data.append([image_path, avg_y, roll_angle])

# Create a new DataFrame with the processed data
new_df = pd.DataFrame(processed_data, columns=['image', 'avg_y', 'roll_angle'])

# Save the processed dataset
new_df.to_csv('horizon_data.csv', index=False)

print(f"Processed {len(new_df)} images")
print("Sample of the processed data:")
print(new_df.head())
