import argparse
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_image


def main():
    parser = argparse.ArgumentParser(description='AI-Powered Medical Image Analysis System')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict'], required=True)
    parser.add_argument('--image', type=str, help='Path to image for prediction mode')
    args = parser.parse_args()

    if args.mode == 'train':
        train_model()
    elif args.mode == 'evaluate':
        evaluate_model()
    elif args.mode == 'predict':
        if not args.image:
            raise ValueError('Please provide --image path for prediction mode.')
        result = predict_image(args.image)
        print(result)


if __name__ == '__main__':
    main()
