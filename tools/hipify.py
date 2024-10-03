import asyncio
import click
import os

async def hipify_file(input_file: str, output_file: str) -> None:
    command = f"hipify-perl {input_file} -o {output_file}"
    print("command: ",command)
    proc = await asyncio.create_subprocess_shell(command)
    await proc.wait()

async def hipify(input_dir: str, output_dir: str) -> None:
    """
    Hipify CUDA scripts in a directory using hipify-perl with asyncio.
    """

    if not output_dir:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".cu"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename.replace(".cu",".hip"))
            tasks.append(hipify_file(input_file, output_file))

    await asyncio.gather(*tasks)

@click.command()
@click.option('--input-dir', '-i', required=True, type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path())
def main(input_dir: str, output_dir: str) -> None:
    """
    Hipify CUDA scripts in a directory using hipify-perl.
    """
    asyncio.run(hipify(input_dir, output_dir))

if __name__ == '__main__':
    main()
