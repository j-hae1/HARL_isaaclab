USING_ISSAC_LAB = False
try:
    # switch to creating Issaclab enviroments instead of default ones
    import omni
    USING_ISSAC_LAB = True
    print("Using Issac Lab")
except ImportError:
    pass

if USING_ISSAC_LAB:
    from omni.isaac.lab_tasks.utils.env_tools.harl_env_tools import (
        make_eval_env,
        make_train_env,
        make_render_env,
        set_seed,
        get_num_agents,
    )
else:
    from harl.utils.envs_tools import (
        make_eval_env,
        make_train_env,
        make_render_env,
        set_seed,
        get_num_agents,
    )