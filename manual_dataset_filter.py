import os
import shutil


def delete_small_folders(root_dir, min_file_count, min_folders_left):
    deleted_folders_count = 0  # Initialize counter for deleted folders

    # Traverse through the directory tree
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # Count the number of files in the directory
            file_count = sum([len(files) for r, d, files in os.walk(dir_path)])
            # If the number of files is less than the specified minimum, delete the directory
            if file_count < min_file_count:
                print(f"Deleting folder: {dir_path} (contains {file_count} files)")
                shutil.rmtree(dir_path)
                deleted_folders_count += 1  # Increment counter

                # Check the number of remaining folders in the root directory
                remaining_folders = sum([len(dirs) for r, dirs, f in os.walk(root_dir)])
                if remaining_folders <= min_folders_left:
                    print(
                        f"Stopped deleting as remaining folders ({remaining_folders}) are less than or equal to the limit ({min_folders_left}).")
                    print(f"Total number of deleted folders: {deleted_folders_count}")
                    return

    print(f"Total number of deleted folders: {deleted_folders_count}")


if __name__ == '__main__':
    delete_small_folders(root_dir='dataset/casia-webface', min_file_count=105, min_folders_left=700)
