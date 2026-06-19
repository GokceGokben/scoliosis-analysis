import os

# Labels directory path
LABELS_DIR = r'[YOUR_LABELS_PATH]'  

def fix_labels(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    parts = line.split()
                    # First row is the class ID, we change it to 0 (vertebra)
                    parts[0] = '0' 
                    new_lines.append(" ".join(parts))
                
                with open(file_path, 'w') as f:
                    f.write("\n".join(new_lines))
    print("All labels are updated to class ID 0.")

fix_labels(LABELS_DIR)