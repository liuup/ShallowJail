import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-model_path",  type=str,   default="./models/Qwen3-4B-Instruct-2507",        help="Victim model path")
    parser.add_argument("-guard_path",  type=str,   default="./models/Qwen3Guard-Gen-4B",        help="Guard model path")
    parser.add_argument("-prompt_path", type=str,   default="./data/forbidden_question_set.txt",    help="jailbreak prompt path")

    parser.add_argument("-alpha",       type=float, default=0,                                      help="Alpha value")
    parser.add_argument("-pre_tokens",  type=int,   default=0,                                      help="Tokens to modify")
    parser.add_argument("-beta",        type=float, default=0,                                      help="Beta value")

    parser.add_argument("-max_new_tokens",       type=int,   default=300,                                      help="max new tokens")

    args = parser.parse_args()

    return args