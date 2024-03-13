import logging, os, sys
from camel.typing import ModelType

root = os.path.dirname(__file__)
sys.path.append(root)

from chatdev.chat_chain import ChatChain

try:
    openai_new_api = True  # new openai api version
except ImportError:
    openai_new_api = False  # old openai api version
    print(
        "Warning: Your OpenAI version is outdated. \n "
        "Please update as specified in requirement.txt. \n "
        "The old API interface is deprecated and will no longer be supported.")


def get_config(company):
    """
    return configuration json files for ChatChain
    user can customize only parts of configuration json files, other files will be left for default
    Args:
        company: customized configuration name under CompanyConfig/

    Returns:
        path to three configuration jsons: [config_path, config_phase_path, config_role_path]
    """
    config_dir = os.path.join(root, "CompanyConfig", company)
    default_config_dir = os.path.join(root, "CompanyConfig", "Default")

    config_files = [
        "ChatChainConfig.json",
        "PhaseConfig.json",
        "RoleConfig.json"
    ]

    config_paths = []

    for config_file in config_files:
        company_config_path = os.path.join(config_dir, config_file)
        default_config_path = os.path.join(default_config_dir, config_file)

        if os.path.exists(company_config_path):
            config_paths.append(company_config_path)
        else:
            config_paths.append(default_config_path)

    return tuple(config_paths)

args = {
    'config': 'Default',
    'org': 'Brass Synregy',
    'name': 'Discord Job Bot',
    'model': 'CLAUDE_OPUS',
    'task': r"""    
                            
""",
    'path': r"",
}

args2type = {
    'CLAUDE_OPUS': ModelType.CLAUDE_3_OPUS,
    'CLAUDE_2': ModelType.CLAUDE_2_1,
}

if 'openai_new_api' in globals() or 'openai_new_api' in locals():
    if openai_new_api:
        args2type['GPT_3_5_TURBO'] = ModelType.CLAUDE_3_OPUS

config_path, config_phase_path, config_role_path = get_config(args['config'])

chat_chain = ChatChain(
    config_path=config_path,
    config_phase_path=config_phase_path,
    config_role_path=config_role_path,
    task_prompt=args['task'],
    project_name=args['name'],
    org_name=args['org'],
    model_type=args2type[args['model']],
    code_path=args['path'],
)

logging.basicConfig(filename=chat_chain.log_filepath, level=logging.INFO,
                    format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', encoding="utf-8")

chat_chain.pre_processing()

chat_chain.make_recruitment()

chat_chain.execute_chain()

chat_chain.post_processing()
