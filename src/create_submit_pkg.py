import datetime
import glob
import subprocess
from zipfile import ZipFile


def create_submit_pkg():

    # Source files
    src_files = glob.glob("src/*.py")

    # Notebooks
    notebooks = glob.glob("*.ipynb")

    # Genereate HTML files from the notebooks
    for nb in notebooks:
        cmd_line = f"jupyter nbconvert --to html {nb}"

        print(f"executing: {cmd_line}")
        subprocess.check_call(cmd_line, shell=True)

    html_files = glob.glob("*.htm*")

    now = datetime.datetime.today().isoformat(timespec="minutes").replace(":", "h")+"m"
    outfile = f"submission_{now}.zip"
    print(f"Adding files to {outfile}")
    with ZipFile(outfile, "w") as zip_object:
        for name in (src_files + notebooks + html_files):
            print(name)
            zip_object.write(name)

    print("")
    msg = f"Done. Please submit the file {outfile}"
    print("-" * len(msg))
    print(msg)
    print("-" * len(msg))


if __name__ == "__main__":
    create_submit_pkg()