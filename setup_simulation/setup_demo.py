import os

if __name__ == "__main__":
    os.makedirs("data/runtime/incoming", exist_ok=True)
    os.makedirs("data/runtime/outgoing", exist_ok=True)
    os.makedirs("data/runtime/processed", exist_ok=True)
