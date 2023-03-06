from jinja2 import Environment, FileSystemLoader, meta
import argparse
from pathlib import Path
import re


def regex_replace(s, find, replace):
    """A non-optimal implementation of a regex filter"""
    return re.sub(find, replace, s)


def main():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--template-file', type=Path, required=True)
    args, unknown = parser.parse_known_args()

    env = Environment(
        loader=FileSystemLoader(args.template_file.parent),
        comment_start_string='{=',  # avoiding env from recognizing ${#var[@]} as a comment
        comment_end_string='=}',
    )
    env.filters['regex_replace'] = regex_replace
    jinja_vars = get_all_jinja_vars(env, args.shell_file.name)
    parser = argparse.ArgumentParser(description='')
    for jinja_var in jinja_vars:
        parser.add_argument(f'--{jinja_var}', type=str, required=True)
    jinja_args = parser.parse_args(unknown)

    template = env.get_template(args.template_file.name)
    output = template.render(vars(jinja_args))
    print(output)


def get_all_jinja_vars(env, filename:str) -> set:
    return meta.find_undeclared_variables(env.parse(env.loader.get_source(env, filename)))


if __name__ == '__main__':
    main()
