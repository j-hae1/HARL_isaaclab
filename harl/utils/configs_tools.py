"""Tools for loading and updating configs."""
import time
import os
import json
import yaml
from uu import Error
import wandb


# TODO: CHANGED
class CompatibleLogger:
    def __init__(self, use_wandb, log_path):
        self.use_wandb = use_wandb

        if not use_wandb:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(log_path)
        else:
            self.writer = None  # wandb handles its own logging separately

    def add_scalar(self, key, value, step):
        if self.use_wandb:
            import wandb
            wandb.log({key: value}, step=step)
        else:
            self.writer.add_scalar(key, value, step)

    def add_scalars(self, main_tag, tag_scalar_dict, step):
        if self.use_wandb:
            import wandb
            # Flatten under main_tag prefix
            wandb.log({f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}, step=step)
        else:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def close(self):
        if not self.use_wandb and self.writer is not None:
            self.writer.export_scalars_to_json(os.path.join(self.writer.log_dir, "summary.json"))
            self.writer.close()

def get_defaults_yaml_args(algo, env):
    """Load config file for user-specified algo and env.
    Args:
        algo: (str) Algorithm name.
        env: (str) Environment name.
    Returns:
        algo_args: (dict) Algorithm config.
        env_args: (dict) Environment config.
    """
    base_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    algo_cfg_path = os.path.join(base_path, "configs", "algos_cfgs", f"{algo}.yaml")
    env_cfg_path = os.path.join(base_path, "configs", "envs_cfgs", f"{env}.yaml")

    with open(algo_cfg_path, "r", encoding="utf-8") as file:
        algo_args = yaml.load(file, Loader=yaml.FullLoader)
    with open(env_cfg_path, "r", encoding="utf-8") as file:
        env_args = yaml.load(file, Loader=yaml.FullLoader)
    return algo_args, env_args


def update_args(unparsed_dict, *args):
    """Update loaded config with unparsed command-line arguments.
    Args:
        unparsed_dict: (dict) Unparsed command-line arguments.
        *args: (list[dict]) argument dicts to be updated.
    """

    def update_dict(dict1, dict2):
        for k in dict2:
            if type(dict2[k]) is dict:
                update_dict(dict1, dict2[k])
            else:
                if k in dict1:
                    dict2[k] = dict1[k]

    for args_dict in args:
        update_dict(unparsed_dict, args_dict)


def get_task_name(env, env_args):
    """Get task name."""
    if env == "smac":
        task = env_args["map_name"]
    elif env == "smacv2":
        task = env_args["map_name"]
    elif env == "mamujoco":
        task = f"{env_args['scenario']}-{env_args['agent_conf']}"
    elif env == "pettingzoo_mpe":
        if env_args["continuous_actions"]:
            task = f"{env_args['scenario']}-continuous"
        else:
            task = f"{env_args['scenario']}-discrete"
    elif env == "gym":
        task = env_args["scenario"]
    elif env == "football":
        task = env_args["env_name"]
    elif env == "dexhands":
        task = env_args["task"]
    elif env == "lag":
        task = f"{env_args['scenario']}-{env_args['task']}"
    elif env == "isaaclab":
        task = env_args["task"]
    return task

# TODO: Changed
def init_dir(env, env_args, algo, seed, logger_path, wandb_kwargs, use_wandb=True):
    """Init directory for saving results."""
    task = get_task_name(env, env_args)
    hms_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    experiment_name = f"{hms_time}_{algo}"
    log_root_path = os.path.join("logs", "harl", logger_path)
    log_root_path = os.path.abspath(log_root_path)
    
    results_path = os.path.join(log_root_path, experiment_name)
    log_path = os.path.join(results_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    models_path = os.path.join(results_path, "checkpoints")
    os.makedirs(models_path, exist_ok=True)

    if use_wandb:
        wandb_kwargs.setdefault("name", experiment_name)
        wandb.init(**wandb_kwargs)
    
    writter = CompatibleLogger(use_wandb=use_wandb, log_path=log_path)
    return results_path, log_path, models_path, writter


def is_json_serializable(value):
    """Check if v is JSON serializable."""
    try:
        json.dumps(value)
        return True
    except Error:
        return False


def convert_json(obj):
    """Convert obj to a version which can be serialized with JSON."""
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, "__name__") and not ("lambda" in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v) for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}

        return str(obj)


def save_config(args, algo_args, env_args, run_dir):
    """Save the configuration of the program."""

    if args['env'] != 'isaaclab':
        config = {"main_args": args, "algo_args": algo_args, "env_args": env_args}
        config_json = convert_json(config)
        output = json.dumps(config_json, separators=(",", ":\t"), indent=4, sort_keys=True)
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as out:
            out.write(output)
