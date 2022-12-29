from utils.parse_args import argument_parser, process_args
from trainer.episodic_trainer import EpisodicTrainer

if __name__ == '__main__':
    args = argument_parser()
    args = process_args(args)
    trainer = EpisodicTrainer(args)
    trainer.train()
    trainer.test()
    trainer.finish()