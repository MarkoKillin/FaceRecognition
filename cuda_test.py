import torch


# This project is capable of training on cuda cores,
# if they are available it will use them,
# if they are not it will train on cpu.
def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        print("CUDA is not available.")


if __name__ == "__main__":
    check_cuda()
