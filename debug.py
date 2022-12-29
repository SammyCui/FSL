from utils.parse_args import DebugArgs
from trainer.episodic_trainer import EpisodicTrainer

if __name__ == '__main__':
    debug_args = DebugArgs(
        model='ProtoNet',
        backbone='convnet',
        dataset_name='omniglot',
        n_ways_train=5,
        n_shots_train=1,
        n_queries_train=15,
        n_ways_test=5,
        n_shots_test=1,
        n_queries_test=15,
        root='./data/data',
        episodes_per_epoch=10,
        temperature=1,
        start_epoch=0,
        max_epoch=5,
        num_val_episodes=15,
        num_test_episodes=20,
        lr=0.001,
        optimizer='sgd',
        lr_scheduler='step',
        step_size=20,
        gamma=0.2,
        momentum=0.9,
        weight_decay=0.0005,
        val_interval=1,
        num_workers=0,
        download=False,
        device='cpu',
        result_dir='./checkpoints/run1')

    trainer = EpisodicTrainer(debug_args)
    trainer.train()
    trainer.test()
    trainer.finish()