
import click
from sacking.environment import load_env
from sacking.policy import GaussianPolicy
from sacking.trainer import simulate
from sacking.typing import Checkpoint


@click.command()
@click.argument('checkpoint', type=click.Path(exists=True))
def main(checkpoint):
    env = load_env('gym/Pendulum-v0')
    checkpoint = Checkpoint.load(checkpoint)
    policy = GaussianPolicy.from_checkpoint(checkpoint)

    while True:
        simulate(policy, env)


if __name__ == '__main__':
    main()
