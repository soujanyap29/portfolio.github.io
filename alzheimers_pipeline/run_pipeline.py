from .download_and_prepare_data import prepare_dataset
from .evaluate import main as evaluate_main
from .train import main as train_main


def main():
    prepare_dataset()
    train_main()
    evaluate_main()


if __name__ == "__main__":
    main()

