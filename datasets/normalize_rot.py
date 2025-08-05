import csv
import math

file_path = input("Enter the path to the CSV file: ")

updated_rows = []
with open(file_path, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    fieldnames = reader.fieldnames
    for row in reader:
        original_rotation = float(row['rotation_radians'])
        if float(row['rotation_radians']) < -math.pi:
            row['rotation_radians'] = str(float(row['rotation_radians']) + 2 * math.pi)
            print(f"Updated rotation for {row['filename']}: {original_rotation} -> {row['rotation_radians']}")
            # write
        elif float(row['rotation_radians']) > math.pi:
            row['rotation_radians'] = str(float(row['rotation_radians']) - 2 * math.pi)
            print(f"Updated rotation for {row['filename']}: {original_rotation} -> {row['rotation_radians']}")
        updated_rows.append(row)


# Write back updated data (overwrite file)
with open(file_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(updated_rows)
    print("CSV updated successfully.")
