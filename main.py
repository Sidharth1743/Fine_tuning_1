def main():
    print("Hello from finetune!")


if __name__ == "__main__":
    main()

import torch
print(torch.__version__)

import torch
print(f"CUDA Available: {torch.cuda.is_available()}")