import subprocess
import sys


def check_and_install_dependencies():
    try:
        # Check for 'sox'
        if sys.platform == "win32":
            subprocess.run(["where", "sox"], check=True)  # Windows-specific
        else:
            subprocess.run(["which", "sox"], check=True)  # Linux/macOS-specific
    except subprocess.CalledProcessError:
        print("sox not found, installing...")
        if sys.platform == "darwin":
            subprocess.run(["brew", "install", "sox"], check=True)
        elif sys.platform in ["linux", "linux2"]:
            subprocess.run(["sudo", "apt-get", "install", "sox"], check=True)
        else:
            print("Please install sox manually. Instructions: https://sox.sourceforge.io/")

    try:
        # Check for 'ffmpeg'
        if sys.platform == "win32":
            subprocess.run(["where", "ffmpeg"], check=True)  # Windows-specific
        else:
            subprocess.run(["which", "ffmpeg"], check=True)  # Linux/macOS-specific
    except subprocess.CalledProcessError:
        print("ffmpeg not found, installing...")
        if sys.platform == "darwin":
            subprocess.run(["brew", "install", "ffmpeg"], check=True)
        elif sys.platform in ["linux", "linux2"]:
            subprocess.run(["sudo", "apt-get", "install", "ffmpeg"], check=True)
        else:
            print("Please install ffmpeg manually. Instructions: https://ffmpeg.org/")


if __name__ == "__main__":
    check_and_install_dependencies()
