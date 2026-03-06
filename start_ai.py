from poppy_ai.config import load_config
from poppy_ai.trainer import Trainer


def main() -> None:
    cfg = load_config("config.json")
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
