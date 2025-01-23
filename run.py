import yaml
import subprocess

from src.common.path_utils import resolve_path

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def execute_steps(steps, single_step = None):
    for step in steps:
        for key, details in step.items():
            name = details.get('name')
            enabled = details.get('enabled', False)
            script = details.get('script')
            arguments = details.get('arguments', {})

            if single_step != None and key != single_step:
                continue
            script_path = script.replace("/",".").lstrip(".").rstrip(".py") #resolve_path(script)

            print(f"[{name}] Running...")
            if enabled or key == single_step:
                cmd = ["./.venv/Scripts/python.exe","-m",script_path]
                for arg_key, arg_value in arguments.items():
                    if isinstance(arg_value, dict):
                        for sub_key, sub_value in arg_value.items():
                            cmd.append(f"--{sub_key}")
                            cmd.append(str(sub_value))
                    else:
                        cmd.append(f"--{arg_key}")
                        cmd.append(str(arg_value))
                try:
                    print(f"Executing command: {" ".join(cmd)}")
                    result = subprocess.run(cmd, check=True, text=True, stderr=subprocess.PIPE)
                    print(f"[{name}] Finished successfully.")
                except subprocess.CalledProcessError as e:
                    print(f"[{name}] Error occured :(\n{e.stderr}")
                    return
            else:
                print(f"[{name}] Step is disabled. Skipping....")

if __name__ == "__main__":
    yaml_file_path = "pipeline.yml"

    try:
        yaml_content = read_yaml(yaml_file_path)
        mode = yaml_content.get('mode', 'sequential')
        steps = yaml_content.get('steps', [])

        if mode == 'single':
            step = yaml_content.get('step', None)
            if step:
                execute_steps(steps, step)
            else:
                print("Error: 'step' key is missing in the YAML file.")
        else:
            execute_steps(steps)

    except Exception as e:
        print(f"Error: {e}")
