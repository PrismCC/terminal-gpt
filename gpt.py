import json
import datetime
from pathlib import Path

from openai import OpenAI


class ColorManager:
    ANSI_RED = "\033[91m"
    ANSI_GREEN = "\033[92m"
    ANSI_YELLOW = "\033[93m"
    ANSI_BLUE = "\033[94m"
    ANSI_MAGENTA = "\033[95m"
    ANSI_CYAN = "\033[96m"
    ANSI_RESET = "\033[0m"

    @classmethod
    def red(cls, text):
        return cls.ANSI_RED + text + cls.ANSI_RESET

    @classmethod
    def green(cls, text):
        return cls.ANSI_GREEN + text + cls.ANSI_RESET

    @classmethod
    def yellow(cls, text):
        return cls.ANSI_YELLOW + text + cls.ANSI_RESET

    @classmethod
    def blue(cls, text):
        return cls.ANSI_BLUE + text + cls.ANSI_RESET

    @classmethod
    def magenta(cls, text):
        return cls.ANSI_MAGENTA + text + cls.ANSI_RESET

    @classmethod
    def cyan(cls, text):
        return cls.ANSI_CYAN + text + cls.ANSI_RESET

    @classmethod
    def red_flag(cls):
        return cls.ANSI_RED

    @classmethod
    def green_flag(cls):
        return cls.ANSI_GREEN

    @classmethod
    def yellow_flag(cls):
        return cls.ANSI_YELLOW

    @classmethod
    def blue_flag(cls):
        return cls.ANSI_BLUE

    @classmethod
    def magenta_flag(cls):
        return cls.ANSI_MAGENTA

    @classmethod
    def cyan_flag(cls):
        return cls.ANSI_CYAN

    @classmethod
    def reset_flag(cls):
        return cls.ANSI_RESET


class Dialog:
    def __init__(self, ask, response):
        self.ask = ask
        self.response = response
        self.time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Message:
    def __init__(self):
        self.messages = []
        self.messages.append(
            {
                "role": "system",
                "content": "你是我的AI助手，将对我的提问进行精简明晰且完整的回答。",
            }
        )

    def add_dialog(self, dialog):
        self.messages.append(
            {
                "role": "user",
                "content": dialog.ask,
            }
        )
        self.messages.append(
            {
                "role": "assistant",
                "content": dialog.response,
            }
        )

    def add_ask(self, ask):
        self.messages.append(
            {
                "role": "user",
                "content": ask,
            }
        )

    def add_instruction(self, instruction):
        self.messages.append(
            {
                "role": "system",
                "content": instruction,
            }
        )

    def get_messages(self):
        return self.messages


class Log:
    def __init__(self, name, dir_path):
        self.name = name
        self.dir = dir_path
        self.path = f"{dir_path}/{name}.log"
        self.data = []
        self.num = 0
        self.pointer = 0

    def save(self):
        dir_path = Path(self.dir)
        if not dir_path.exists():
            dir_path.mkdir(parents=True)

        with open(self.path, "a", encoding="utf-8") as f:
            for i in range(self.pointer, self.num):
                dialog = self.data[i]
                f.write(
                    f"{dialog.time}\n\nUser:\n{dialog.ask}\n\n{self.name}:\n{dialog.response}\n\n--------------------------------\n\n"
                )
        self.pointer = self.num

    def reset(self):
        self.save()
        self.data = []
        self.num = 0
        self.pointer = 0

    def clean(self):
        self.data = []
        self.num = 0
        self.pointer = 0
        with open(self.path, "w") as f:
            pass

    def append(self, dialog):
        self.data.append(dialog)
        self.num += 1


def load_config():
    with open("config.json", "r") as f:
        config = json.load(f)
    if config is None:
        return None
    return config


def save_config(config: dict):
    json_data = json.dumps(config, indent=4)
    with open("config.json", "w") as f:
        f.write(json_data)


def init_cilent(config: dict):
    return OpenAI(
        api_key=config["api_key"],
        base_url=config["url"],
    )


def generate_message(log: Log, ask: str, context_len: int, instruction: str = ""):
    message = Message()
    if instruction:
        message.add_instruction(instruction)
    start_index = log.num - context_len if log.num > context_len else 0
    for i in range(start_index, log.num):
        dialog = log.data[i]
        message.add_dialog(dialog)
    message.add_ask(ask)
    return message.get_messages()


def create_stream(client: OpenAI, config: dict, messages: list):
    return client.chat.completions.create(
        model=config["model"],
        messages=messages,
        stream=True,
    )


def print_stream(stream):
    response = ""
    for chunk in stream:
        if (
            chunk.choices
            and chunk.choices[0]
            and chunk.choices[0].delta
            and chunk.choices[0].delta.content
        ):
            print(ColorManager.blue(chunk.choices[0].delta.content), end="")
            response += chunk.choices[0].delta.content
    return response


def print_config(config: dict):
    for k, v in config.items():
        if k != "api_key" and k != "url":
            print(ColorManager.cyan(f"{k}: {v}"))


def main():
    config = load_config()
    if config is None:
        print("Please configure the config first.")
        return

    client = init_cilent(config)

    model_name = config["model"]
    logs_path = "./logs"
    log = Log(model_name, logs_path)
    in_path = "./in.txt"

    instruction = ""

    print_config(config)

    while True:
        ask = input("> ").strip()
        func = ask.split(" ")[0].lower()
        para = ask.split(" ")[1] if ask.count(" ") > 0 else None

        if func == "e" or func == "exit":
            print(ColorManager.yellow("Bye!"))
            break
        elif func == "r" or func == "reset":
            log.reset()
            instruction = ""
            print(ColorManager.yellow("Dialogs are reset."))
            continue
        elif func == "c" or func == "clean":
            log.clean()
            print(ColorManager.yellow("Logs are cleaned."))
            continue
        elif func == "m" or func == "model":
            if para is None:
                print(ColorManager.yellow("Please input model name."))
                continue

            if para in "gpt-4o-mini":
                para = "gpt-4o-mini"
            elif para in "gpt-4o":
                para = "gpt-4o"
            elif para in "gpt-4-turbo":
                para = "gpt-4-turbo"

            config["model"] = para
            model_name = para
            log_path = f"logs/{model_name}.log"
            log = Log(model_name, log_path)
            save_config(config)
            print(ColorManager.yellow(f"Model is changed to {model_name}."))
            print_config(config)
            continue
        elif func == "l" or func == "len":
            config["context_len"] = int(para)
            save_config(config)
            print(ColorManager.yellow(f"Context length is changed to {para}."))
            print_config(config)
            continue
        elif func == "f" or func == "file":
            with open(in_path, "r", encoding="utf-8") as f:
                ask = f.read()
            print(ColorManager.green(ask))
            ans = input(ColorManager.yellow("\nask these? (y/n)")).strip()
            if ans != "y" and ans != "Y":
                print(ColorManager.yellow("Canceled.\n"))
                continue
        elif func == "i" or func == "ins":
            instruction = input(ColorManager.yellow("Instruction: ")).strip()
            print(ColorManager.yellow("Instruction is set.\n"))
            continue
        elif func == "h" or func == "help":
            print(ColorManager.magenta("Commands:"))
            print(ColorManager.magenta("  e/exit                     :  Exit"))
            print(ColorManager.magenta("  r/reset                    :  Reset dialogs"))
            print(ColorManager.magenta("  c/clean                    :  Clean logs"))
            print(ColorManager.magenta("  m/model + [model name]     :  Change model"))
            print(
                ColorManager.magenta(
                    "  l/len   + [context length] :  Change context length"
                )
            )
            print(
                ColorManager.magenta("  f/file  + [file name]      :  Read from file")
            )
            print(
                ColorManager.magenta("  i/ins   + [instruction]    :  Set instruction")
            )
            print(ColorManager.magenta("  h/help                     :  Show help"))
            print(ColorManager.magenta("  [others]                   :  Chat with GPT"))
            print()
            continue

        print("\nUser:\n" + ColorManager.green(ask))

        message = generate_message(log, ask, config["context_len"], instruction)
        stream = create_stream(client, config, message)
        print(f"\n{model_name}:")
        response = print_stream(stream)
        print(ColorManager.yellow("\n--END--\n\n"))
        log.append(Dialog(ask, response))
        log.save()


if __name__ == "__main__":
    main()
