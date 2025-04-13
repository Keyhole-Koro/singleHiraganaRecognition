import os
import re

def is_single_hiragana(char):
    return re.match(r'^[\u3040-\u309F]$', char) is not None

def remove_non_hiragana_data(label_file, image_dir, output_label_file, output_image_dir):
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    with open(label_file, 'r', encoding='utf-8') as infile, open(output_label_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            image_name, label = line.strip().split()
            if is_single_hiragana(label):
                infile_path = os.path.join(image_dir, image_name)
                outfile_path = os.path.join(output_image_dir, image_name)
                if os.path.exists(infile_path):
                    os.rename(infile_path, outfile_path)
                    outfile.write(f"{image_name} {label}\n")
                else:
                    print(f"Warning: Image {infile_path} not found.")

# Example usage
if __name__ == "__main__":
    label_file = 'hiragana/eval/label.txt'  # Replace with the path to your label file
    image_dir = 'hiragana/eval'  # Replace with the path to your image directory
    output_label_file = 'hiragana/eval_filtered/label.txt'  # Replace with the path to your output label file
    output_image_dir = 'hiragana/eval_filtered'  # Replace with the path to your output image directory
    remove_non_hiragana_data(label_file, image_dir, output_label_file, output_image_dir)
