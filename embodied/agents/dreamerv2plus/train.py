import pathlib
import sys
import warnings
import logging

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')

directory = pathlib.Path(__file__)
try:
  import google3  # type: ignore # noqa
except ImportError:
  directory = directory.resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import embodied

def create_logger(logdir: embodied.Path) -> logging.Logger:
  # Logger config
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)
  formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')

  logdir = embodied.Path(logdir)
  logdir.mkdirs()
  log_file = logdir / 'data.log'

  file_handler = logging.FileHandler(log_file)
  file_handler.setLevel(logging.DEBUG)
  file_handler.setFormatter(formatter)

  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.ERROR)
  console_handler.setFormatter(formatter)

  logger.addHandler(file_handler)
  logger.addHandler(console_handler)
  return logger

def main(argv=None):
  from . import agent as agnt

  parsed, other = embodied.Flags(
      configs=['defaults'], worker=0, workers=1,
      learner_addr='localhost:2222',
  ).parse_known(argv)
  config = embodied.Config(agnt.Agent.configs['defaults'])
  for name in parsed.configs:
    config = config.update(agnt.Agent.configs[name])
  config = embodied.Flags(config).parse(other)

  config = config.update(logdir=str(embodied.Path(config.logdir)))
  args = embodied.Config(logdir=config.logdir, **config.train)
  args = args.update(expl_until=args.expl_until // config.env.repeat)

  message_logger = create_logger(embodied.Path(config.logdir))
  message_logger.info(config)

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()

  outdir = logdir
  multiplier = config.env.repeat
  if config.run == 'acting':
    outdir /= f'worker{parsed.worker}'
    multiplier *= parsed.workers
  elif config.run == 'learning':
    outdir /= 'learner'
    multiplier = 1
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(config.filter),
      embodied.logger.JSONLOutput(outdir, 'metrics.jsonl'),
      embodied.logger.JSONLOutput(outdir, 'scores.jsonl', 'episode/score'),
      embodied.logger.TensorBoardOutput(outdir),
  ], multiplier)

  cleanup = []
  try:
    config = config.update({'env.seed': hash((config.seed, parsed.worker))})
    env = embodied.envs.load_env(
        config.task, mode='train', logdir=logdir, **config.env)
    agent = agnt.Agent(env.obs_space, env.act_space, step, config)
    cleanup.append(env)

    if config.run == 'train':
      replay = make_replay(config, logdir / 'episodes')
      embodied.run.train(agent, env, replay, logger, args)

    elif config.run == 'train_eval':
      eval_env = embodied.envs.load_env(
          config.task, mode='eval', logdir=logdir, **config.env)
      replay = make_replay(config, logdir / 'episodes')
      eval_replay = make_replay(config, logdir / 'eval_episodes', is_eval=True)
      embodied.run.train_eval(
          agent, env, eval_env, replay, eval_replay, logger, args)
      cleanup.append(eval_env)

    elif config.run == 'train_fixed_eval':
      if config.eval_dir:
        assert not config.train.eval_fill
        eval_replay = make_replay(config, config.eval_dir, is_eval=True)
      else:
        assert config.train.eval_fill
        eval_replay = make_replay(
            config, logdir / 'eval_episodes', is_eval=True)
      replay = make_replay(config, logdir / 'episodes')
      embodied.run.train_fixed_eval(
          agent, env, replay, eval_replay, logger, args)

    elif config.run == 'learning':
      env.close()
      port = parsed.learner_addr.split(':')[-1]
      replay = make_replay(config, logdir / 'episodes', server_port=port)
      if config.eval_dir:
        eval_replay = make_replay(config, config.eval_dir, is_eval=True)
      else:
        eval_replay = replay
      embodied.run.learning(agent, replay, eval_replay, logger, args)

    elif config.run == 'acting':
      replay = make_replay(config, remote_addr=parsed.learner_addr)
      embodied.run.acting(agent, env, replay, logger, outdir, args)

    elif config.run == "eval":  # TODO: Implement inference
      raise NotImplementedError(config.run)

    else:
      raise NotImplementedError(config.run)
  finally:
    for obj in cleanup:
      obj.close()


def make_replay(
    config, directory=None, is_eval=False, remote_addr=None,
    server_port=None, **kwargs):

  size = config.replay_size
  chunk = config.replay_chunk
  if is_eval:
    size //= 10
  if remote_addr:
    store = embodied.replay.StoreClient(remote_addr)
  else:
    if directory:
      store = embodied.replay.CkptRAMStore(
          directory, size, parallel=True)
    else:
      store = embodied.replay.RAMStore(size)
  store = embodied.replay.Stats(store)
  if server_port:
    store = embodied.replay.StoreServer(store, server_port)

  if config.replay == 'fixed':
    replay = embodied.replay.FixedLength(
        store, chunk, **config.replay_fixed, **kwargs)
  elif config.replay == 'consec':
    replay = embodied.replay.Consecutive(
        store, chunk, **config.replay_consec, **kwargs)
  elif config.replay == 'prio' and not is_eval:
    replay = embodied.replay.Prioritized(
        store, chunk, **config.replay_prio, **kwargs)
  elif config.replay == 'prio' and is_eval:
    # Don't use prioritized for eval.
    replay = embodied.replay.FixedLength(
        store, chunk, **config.replay_fixed, **kwargs)
  else:
    raise NotImplementedError(config.replay)
  return replay


if __name__ == '__main__':
  main()
